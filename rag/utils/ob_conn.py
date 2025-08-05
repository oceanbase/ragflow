#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import os
import re
import time
from typing import Any, Optional

from pydantic import BaseModel
from pymysql.converters import escape_string
from pyobvector import ObVecClient, FtsIndexParam, FtsParser, ARRAY, VECTOR
from pyobvector.schema import ObTable
from pyobvector.util import ObVersion
from sqlalchemy import text, Column, String, Integer, JSON, Double, Row, Table
from sqlalchemy.dialects.mysql import LONGTEXT, TEXT
from sqlalchemy.sql.type_api import TypeEngine

from api.utils import get_base_config
from rag import settings
from rag.settings import PAGERANK_FLD
from rag.utils import singleton, get_float
from rag.utils.doc_store_conn import DocStoreConnection, MatchExpr, OrderByExpr, FusionExpr, MatchTextExpr, \
    MatchDenseExpr
from rag.utils.redis_conn import distributed_lock

ATTEMPT_TIME = 2
OB_QUERY_TIMEOUT = int(os.environ.get("OB_QUERY_TIMEOUT", "100_000_000"))

logger = logging.getLogger('ragflow.ob_conn')

column_definitions: list[Column] = [
    Column("id", String(256), primary_key=True, comment="chunk id"),
    Column("kb_id", String(256), nullable=False, comment="knowledge base id"),
    Column("doc_id", String(256), nullable=True, comment="document id"),
    Column("docnm_kwd", String(256), nullable=True, comment="document name"),
    Column("doc_type_kwd", String(256), nullable=True, comment="document type"),
    Column("title_tks", String(256), nullable=True, comment="title tokens"),
    Column("title_sm_tks", String(256), nullable=True, comment="fine-grained (small) title tokens"),
    Column("content_with_weight", LONGTEXT, nullable=True, comment="the original content"),
    Column("content_ltks", LONGTEXT, nullable=True, comment="long text tokens derived from content_with_weight"),
    Column("content_sm_ltks", LONGTEXT, nullable=True, comment="fine-grained (small) tokens derived from content_ltks"),
    Column("pagerank_fea", Integer, nullable=True, comment="page rank priority, usually set in kb level"),
    Column("important_kwd", ARRAY(String(256)), nullable=True, comment="keywords"),
    Column("important_tks", TEXT, nullable=True, comment="keyword tokens"),
    Column("question_kwd", ARRAY(String(1024)), nullable=True, comment="questions"),
    Column("question_tks", TEXT, nullable=True, comment="question tokens"),
    Column("tag_kwd", ARRAY(String(256)), nullable=True, comment="tags"),
    Column("tag_feas", JSON, nullable=True,
           comment="tag features used for 'rank_feature', format: [tag -> relevance score]"),
    Column("available_int", Integer, nullable=False, server_default="1",
           comment="status of availability, 0 for unavailable, 1 for available"),
    Column("create_time", String(19), nullable=True, comment="creation time in YYYY-MM-DD HH:MM:SS format"),
    Column("create_timestamp_flt", Double, nullable=True, comment="creation timestamp in float format"),
    Column("img_id", String(128), nullable=True, comment="image id"),
    Column("position_int", ARRAY(ARRAY(Integer)), nullable=True, comment="position"),
    Column("page_num_int", ARRAY(Integer), nullable=True, comment="page number"),
    Column("top_int", ARRAY(Integer), nullable=True, comment="rank from the top"),
    Column("knowledge_graph_kwd", String(256), nullable=True, comment="knowledge graph chunk type"),
    Column("source_id", ARRAY(String(256)), nullable=True, comment="source document id"),
    Column("entity_kwd", String(256), nullable=True, comment="entity name"),
    Column("entity_type_kwd", String(256), nullable=True, comment="entity type"),
    Column("from_entity_kwd", String(256), nullable=True, comment="the source entity of this edge"),
    Column("to_entity_kwd", String(256), nullable=True, comment="the target entity of this edge"),
    Column("weight_int", Integer, nullable=True, comment="the weight of this edge"),
    Column("weight_flt", Double, nullable=True, comment="the weight of community report"),
    Column("entities_kwd", ARRAY(String(256)), nullable=True, comment="node ids of entities"),
    Column("rank_flt", Double, nullable=True, comment="rank of this entity"),
    Column("removed_kwd", String(256), nullable=True, server_default="'N'", comment="whether it has been deleted"),
    Column("metadata", JSON, nullable=True, comment="metadata for this chunk"),
    Column("extra", JSON, nullable=True, comment="extra information of non-general chunk"),
    Column("_order_id", Integer, nullable=True, comment="chunk order id for maintaining sequence"),
]

column_names: list[str] = [col.name for col in column_definitions]
column_types: dict[str, TypeEngine] = {col.name: col.type for col in column_definitions}
array_columns: list[str] = [col.name for col in column_definitions if isinstance(col.type, ARRAY)]

vector_column_pattern = re.compile(r"q_(?P<vector_size>\d+)_vec")

fulltext_search_columns: list[str] = [
    "title_tks",
    "title_sm_tks",
    "content_ltks",
    "content_sm_ltks",
    "important_tks",
    "question_tks",
]

fulltext_index_name_template = "fts_idx_%s"
# MATCH AGAINST: https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000002017607
fulltext_search_template = "MATCH (%s) AGAINST ('%s' IN NATURAL LANGUAGE MODE)"
# cosine_distance: https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000002012938
vector_search_template = "cosine_distance(%s, %s)"


class SearchResult(BaseModel):
    total: int
    chunks: list[dict]


def get_column_value(column_name: str, value: Any) -> Any:
    if column_name in column_types:
        column_type = column_types[column_name]
        if isinstance(column_type, String):
            return str(value)
        elif isinstance(column_type, Integer):
            return int(value)
        elif isinstance(column_type, Double):
            return float(value)
        elif isinstance(column_type, ARRAY) or isinstance(column_type, JSON):
            return json.loads(value) if isinstance(value, str) else value
        else:
            raise ValueError(f"Unsupported column type for column '{column_name}': {column_type}")
    elif vector_column_pattern.match(column_name):
        return json.loads(value) if isinstance(value, str) else value
    elif column_name == "_score":
        return float(value)
    else:
        raise ValueError(f"Unknown column '{column_name}' with value '{value}'.")


def get_default_value(column_name: str) -> Any:
    if column_name == "available_int":
        return 1
    elif column_name == "removed_kwd":
        return "N"
    elif column_name == "_order_id":
        return 0
    else:
        return None


def get_value_str(value: Any) -> str:
    if isinstance(value, str):
        return f"'{escape_string(value)}'"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif value is None:
        return "NULL"
    elif isinstance(value, (list, dict)):
        json_str = json.dumps(value, ensure_ascii=False)
        return f"'{escape_string(json_str)}'"
    else:
        return str(value)


def get_filters(condition: dict) -> list[str]:
    filters: list[str] = []
    for k, v in condition.items():
        if not v:
            continue

        if k == "exists":
            filters.append(f"{v} IS NOT NULL")
        elif k == "must_not" and isinstance(v, dict) and "exists" in v:
            filters.append(f"{v.get('exists')} IS NULL")
        elif k in array_columns:
            if isinstance(v, list):
                array_filters = []
                for vv in v:
                    array_filters.append(f"array_contains({k}, {get_value_str(vv)})")
                array_filter = " OR ".join(array_filters)
                filters.append(f"({array_filter})")
            else:
                filters.append(f"array_contains({k}, {get_value_str(v)})")
        elif isinstance(v, list):
            values: list[str] = []
            for item in v:
                values.append(get_value_str(item))
            value = ", ".join(values)
            filters.append(f"{k} IN ({value})")
        else:
            filters.append(f"{k} = {get_value_str(v)}")
    return filters


def _try_with_lock(lock_name: str, process_func, check_func, timeout: int = 5):
    if not check_func():
        lock = distributed_lock(lock_name)
        if lock.acquire():
            logger.info(f"acquired lock success: {lock_name}, start processing.")
            try:
                process_func()
                return
            finally:
                lock.release()

    if not check_func():
        logger.info(f"Waiting for process complete for {lock_name} on other task executors.")
        time.sleep(1)
        count = 1
        while count < timeout and not check_func():
            count += 1
            time.sleep(1)
        if count >= timeout and not check_func():
            raise Exception(f"Timeout to wait for process complete for {lock_name}.")


@singleton
class OBConnection(DocStoreConnection):
    def __init__(self):
        scheme: str = settings.OB.get("scheme")
        ob_config = settings.OB.get("config", {})

        if scheme and scheme.lower() == "mysql":
            mysql_config = get_base_config("mysql", {})
            logger.info("Use MySQL schema to create OceanBase connection.")
            host = mysql_config.get("host", "localhost")
            port = mysql_config.get("port", 2881)
            self.username = mysql_config.get("user", "root@test")
            self.password = mysql_config.get("password", "infini_rag_flow")
        else:
            logger.info("Use customized config to create OceanBase connection.")
            host = ob_config.get("host", "localhost")
            port = ob_config.get("port", 2881)
            self.username = ob_config.get("user", "root@test")
            self.password = ob_config.get("password", "infini_rag_flow")

        self.db_name = ob_config.get("db_name", "infini_rag_flow")
        self.uri = f"{host}:{port}"

        logger.info(f"Use OceanBase '{self.uri}' as the doc engine.")

        for _ in range(ATTEMPT_TIME):
            try:
                self.client = ObVecClient(
                    uri=self.uri,
                    user=self.username,
                    password=self.password,
                    db_name=self.db_name,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting OceanBase {self.uri} to be healthy.")
                time.sleep(5)

        if self.client is None:
            msg = f"OceanBase {self.uri} connection failed after {ATTEMPT_TIME} attempts."
            logger.error(msg)
            raise Exception(msg)

        self._check_ob_version()
        self._try_to_update_ob_query_timeout()

        logger.info(f"OceanBase {self.uri} is healthy.")

    def _check_ob_version(self):
        try:
            res = self.client.perform_raw_text_sql("SELECT OB_VERSION() FROM DUAL").fetchone()
            version_str = res[0] if res else None
        except Exception as e:
            raise Exception(f"Failed to get OceanBase version from {self.uri}, error: {str(e)}")

        if not version_str:
            raise Exception(f"Failed to get OceanBase version from {self.uri}.")

        ob_version = ObVersion.from_db_version_string(version_str)
        if ob_version < ObVersion.from_db_version_nums(4, 3, 5, 1):
            raise Exception(
                f"The version of OceanBase needs to be higher than or equal to 4.3.5.1, current version is {version_str}"
            )

    def _try_to_update_ob_query_timeout(self):
        try:
            val = self._get_variable_value("ob_query_timeout")
            if val and int(val) >= OB_QUERY_TIMEOUT:
                return
        except Exception as e:
            logger.warning("Failed to get 'ob_query_timeout' variable: %s", str(e))

        try:
            self.client.perform_raw_text_sql(f"SET GLOBAL ob_query_timeout={OB_QUERY_TIMEOUT}")
            logger.info("Set GLOBAL variable 'ob_query_timeout' to %d.", OB_QUERY_TIMEOUT)
        except Exception as e:
            logger.warning(f"Failed to set 'ob_query_timeout' variable: {str(e)}")

    """
    Database operations
    """

    def dbType(self) -> str:
        return "oceanbase"

    def health(self) -> dict:
        return {
            "uri": self.uri,
            "version_comment": self._get_variable_value("version_comment")
        }

    def _get_variable_value(self, var_name: str) -> Any:
        rows = self.client.perform_raw_text_sql(f"SHOW VARIABLES LIKE '{var_name}'")
        for row in rows:
            return row[1]
        raise Exception(f"Variable '{var_name}' not found.")

    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        vector_field_name = f"q_{vectorSize}_vec"
        vector_index_name = f"{vector_field_name}_idx"

        try:
            _try_with_lock(
                lock_name=f"ob_create_table_{indexName}",
                check_func=lambda: self.client.check_table_exists(indexName),
                process_func=lambda: self._create_table(indexName),
            )

            for column_name in fulltext_search_columns:
                _try_with_lock(
                    lock_name=f"ob_add_fulltext_idx_{indexName}_{column_name}",
                    check_func=lambda: self._index_exists(indexName, fulltext_index_name_template % column_name),
                    process_func=lambda: self._add_fulltext_index(indexName, column_name),
                )

            _try_with_lock(
                lock_name=f"ob_add_vector_column_{indexName}_{vector_field_name}",
                check_func=lambda: self._column_exist(indexName, vector_field_name),
                process_func=lambda: self._add_vector_column(indexName, vectorSize),
            )

            _try_with_lock(
                lock_name=f"ob_add_vector_idx_{indexName}_{vector_field_name}",
                check_func=lambda: self._index_exists(indexName, vector_index_name),
                process_func=lambda: self._add_vector_index(indexName, vector_field_name),
            )
        except Exception as e:
            raise Exception(f"OBConnection.createIndex error: {str(e)}")
        finally:
            # always refresh metadata to make sure it contains the latest table structure
            self.client.refresh_metadata([indexName])

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        if len(knowledgebaseId) > 0:
            # The index need to be alive after any kb deletion since all kb under this tenant are in one index.
            return
        try:
            if self.client.check_table_exists(table_name=indexName):
                self.client.drop_table_if_exist(indexName)
                logger.info(f"Dropped table '{indexName}'.")
        except Exception as e:
            raise Exception(f"OBConnection.deleteIndex error: {str(e)}")

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        try:
            return self.client.check_table_exists(indexName)
        except Exception as e:
            raise Exception(f"OBConnection.indexExist error: {str(e)}")

    def _get_count(self, table_name: str, filter_list: list[str] = None) -> int:
        where_clause = "WHERE " + " AND ".join(filter_list) if len(filter_list) > 0 else ""
        (count,) = self.client.perform_raw_text_sql(
            f"SELECT COUNT(*) FROM {table_name} {where_clause}"
        ).fetchone()
        return count

    def _column_exist(self, table_name: str, column_name: str) -> bool:
        return self._get_count(
            table_name="INFORMATION_SCHEMA.COLUMNS",
            filter_list=[
                f"TABLE_SCHEMA = '{self.db_name}'",
                f"TABLE_NAME = '{table_name}'",
                f"COLUMN_NAME = '{column_name}'",
            ]) > 0

    def _index_exists(self, table_name: str, index_name: str) -> bool:
        return self._get_count(
            table_name="INFORMATION_SCHEMA.STATISTICS",
            filter_list=[
                f"TABLE_SCHEMA = '{self.db_name}'",
                f"TABLE_NAME = '{table_name}'",
                f"INDEX_NAME = '{index_name}'",
            ]) > 0

    def _create_table(self, table_name: str):
        # remove outdated metadata for external changes
        if table_name in self.client.metadata_obj.tables:
            self.client.metadata_obj.remove(Table(table_name, self.client.metadata_obj))

        table = ObTable(
            table_name,
            self.client.metadata_obj,
            *column_definitions,
            mysql_charset='utf8mb4',
            mysql_collate='utf8mb4_unicode_ci',
        )
        table.create(self.client.engine, checkfirst=True)
        logger.info(f"Created table '{table_name}'.")

    def _add_fulltext_index(self, table_name: str, column_name: str):
        fulltext_index_name = fulltext_index_name_template % column_name
        self.client.create_fts_idx_with_fts_index_param(
            table_name=table_name,
            fts_idx_param=FtsIndexParam(
                index_name=fulltext_index_name,
                field_names=[column_name],
                parser_type=FtsParser.IK,
            ),
        )
        logger.info(f"Created full text index '{fulltext_index_name}' on table '{table_name}'.")

    def _add_vector_column(self, table_name: str, vector_size: int):
        vector_field_name = f"q_{vector_size}_vec"

        self.client.add_columns(
            table_name=table_name,
            columns=[Column(vector_field_name, VECTOR(vector_size), nullable=True)],
        )
        logger.info(f"Added vector column '{vector_field_name}' to table '{table_name}'.")

    def _add_vector_index(self, table_name: str, vector_field_name: str):
        vector_index_name = f"{vector_field_name}_idx"
        self.client.create_index(
            table_name=table_name,
            is_vec_index=True,
            index_name=vector_index_name,
            column_names=[vector_field_name],
            vidx_params="distance=cosine, type=hnsw, lib=vsag",
        )
        logger.info(
            f"Created vector index '{vector_index_name}' on table '{table_name}' with column '{vector_field_name}'."
        )

    """
    CRUD operations
    """

    def search(
            self,
            selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None
    ):
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0
        indexNames = list(set(indexNames))

        output_fields = selectFields.copy()
        if "id" not in output_fields:
            output_fields = ["id"] + output_fields
        if "_score" in output_fields:
            output_fields.remove("_score")

        if highlightFields:
            for field in highlightFields:
                if field not in output_fields:
                    output_fields.append(field)

        condition["kb_id"] = knowledgebaseIds
        filters: list[str] = get_filters(condition)
        filters_expr = " AND ".join(filters)

        fulltext_query: Optional[str] = None
        fulltext_topn: Optional[int] = None
        fulltext_search_weight: dict[str, float] = {}
        fulltext_search_expr: dict[str, str] = {}

        vector_column_name: Optional[str] = None
        vector_data: Optional[list[float]] = None
        vector_topn: Optional[int] = None
        vector_similarity_threshold: Optional[float] = None
        vector_similarity_weight: Optional[float] = None

        if len(matchExprs) == 3 and os.getenv('DISABLE_FULLTEXT', 'false').lower() in ['true', '1', 'yes', 'y']:
            # disable fulltext search in fusion search, which means fallback to vector search
            matchExprs = [m for m in matchExprs if isinstance(m, MatchDenseExpr)]

        for m in matchExprs:
            if isinstance(m, MatchTextExpr):
                assert "origin_keywords" in m.extra_options, f"'origin_keywords' is missing in extra_options."
                fulltext_query = m.extra_options["origin_keywords"]
                if isinstance(fulltext_query, list):
                    fulltext_query = " ".join(fulltext_query)
                fulltext_query = escape_string(fulltext_query.strip())
                fulltext_topn = m.topn

                # get fulltext match expression and weight values
                for field in m.fields:
                    parts = field.split("^")
                    column_name: str = parts[0]
                    column_weight: float = float(parts[1]) if (len(parts) > 1 and parts[1]) else 1.0

                    if column_name in fulltext_search_columns:
                        fulltext_search_weight[column_name] = column_weight
                        fulltext_search_expr[column_name] = fulltext_search_template % (column_name, fulltext_query)

                # adjust the weight to 0~1
                weight_sum = sum(fulltext_search_weight.values())
                for column_name in fulltext_search_weight.keys():
                    fulltext_search_weight[column_name] = fulltext_search_weight[column_name] / weight_sum

            elif isinstance(m, MatchDenseExpr):
                assert m.embedding_data_type == "float", f"embedding data type '{m.embedding_data_type}' is not float."
                vector_column_name = m.vector_column_name
                vector_data = m.embedding_data
                vector_topn = m.topn
                vector_similarity_threshold = m.extra_options.get("similarity", 0.0)
            elif isinstance(m, FusionExpr):
                weights = m.fusion_params["weights"]
                vector_similarity_weight = get_float(weights.split(",")[1])

        if fulltext_query:
            fulltext_match_filter = f"({' OR '.join([expr for expr in fulltext_search_expr.values()])})"
            fulltext_match_score_expr = f"({' + '.join(f'{expr} * {fulltext_search_weight.get(col, 0)}' for col, expr in fulltext_search_expr.items())})"

        vector_distance_expr = vector_search_template % (vector_column_name, vector_data) if vector_data else None

        # TODO use tag rank_feature in sorting
        tag_rank_fea = {k: float(v) for k, v in (rank_feature or {}).items() if k != PAGERANK_FLD}

        result: SearchResult = SearchResult(
            total=0,
            chunks=[],
        )
        for index_name in indexNames:

            if not self.client.check_table_exists(index_name):
                continue

            if fulltext_query and vector_distance_expr:
                # fusion search, usually for chat
                count_sql = (
                    f"WITH fulltext_results AS ("
                    f"  SELECT id FROM {index_name}"
                    f"      WHERE {filters_expr} AND {fulltext_match_filter}"
                    f"      LIMIT {fulltext_topn}"
                    f"),"
                    f"vector_results AS ("
                    f"  SELECT id FROM {index_name}"
                    f"      WHERE {filters_expr} AND {vector_column_name} IS NOT NULL AND (1 - {vector_distance_expr}) > {vector_similarity_threshold}"
                    f"      ORDER BY {vector_distance_expr}"
                    f"      APPROXIMATE LIMIT {vector_topn}"
                    f")"
                    f"  SELECT COUNT(*) FROM fulltext_results f FULL OUTER JOIN vector_results v ON f.id = v.id"
                )
                logger.debug("OBConnection.search with count sql: %s", count_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(count_sql)
                total_count = res.fetchone()[0] if res else 0
                result.total += total_count

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: fusion, step: 1-count, elapsed time: {elapsed_time:.3f} seconds,"
                    f" vector column: '{vector_column_name}',"
                    f" query text: '{fulltext_query}',"
                    f" condition: '{condition}',"
                    f" got count: {total_count}"
                )

                if total_count == 0:
                    continue

                assert vector_similarity_weight is not None, "vector_similarity_weight must be set for fusion search."
                fields_expr = ", ".join([f"t.{f} as {f}" for f in output_fields if f != "_score"])
                if "_score" not in output_fields:
                    output_fields.append("_score")
                score_expr_list = [
                    f"(1 - {vector_similarity_weight}) * COALESCE(f.relevance, 0)",
                    f"{vector_similarity_weight} * COALESCE(v.similarity, 0)",
                    "(CAST(IFNULL(COALESCE(f.pagerank_fea, v.pagerank_fea), 0) AS DECIMAL(10, 2)) / 100)",
                ]
                score_expr = f"({' + '.join(score_expr_list)})"
                fusion_sql = (
                    f"WITH fulltext_results AS ("
                    f"  SELECT id, pagerank_fea, {fulltext_match_score_expr} AS relevance"
                    f"      FROM {index_name}"
                    f"      WHERE {filters_expr} AND {fulltext_match_filter}"
                    f"      LIMIT {fulltext_topn}"
                    f"),"
                    f"vector_results AS ("
                    f"  SELECT id, pagerank_fea, (1 - {vector_distance_expr}) AS similarity"
                    f"      FROM {index_name}"
                    f"      WHERE {filters_expr} AND {vector_column_name} IS NOT NULL AND (1 - {vector_distance_expr}) > {vector_similarity_threshold}"
                    f"      ORDER BY {vector_distance_expr}"
                    f"      APPROXIMATE LIMIT {vector_topn}"
                    f"),"
                    f"combined_results AS ("
                    f"  SELECT COALESCE(f.id, v.id) AS id, {score_expr} AS score"
                    f"      FROM fulltext_results f"
                    f"      FULL OUTER JOIN vector_results v"
                    f"      ON f.id = v.id"
                    f")"
                    f"  SELECT {fields_expr}, c.score as _score"
                    f"      FROM combined_results c"
                    f"      JOIN {index_name} t"
                    f"      ON c.id = t.id"
                    f"      ORDER BY score DESC"
                    f"      LIMIT {offset}, {limit}"
                )
                logger.debug("OBConnection.search with fusion sql: %s", fusion_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(fusion_sql)
                rows = res.fetchall()

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: fusion, step: 2-query, elapsed time: {elapsed_time:.3f} seconds,"
                    f" select fields: '{output_fields}',"
                    f" vector column: '{vector_column_name}',"
                    f" query text: '{fulltext_query}',"
                    f" condition: '{condition}',"
                    f" return rows count: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))
            elif vector_distance_expr:
                # vector search, usually used for graph search
                count_sql = (
                    f"SELECT COUNT(id) FROM {index_name}"
                    f"  WHERE {filters_expr} AND {vector_column_name} IS NOT NULL AND (1 - {vector_distance_expr}) > {vector_similarity_threshold}"
                )
                logger.debug("OBConnection.search with vector count sql: %s", count_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(count_sql)
                total_count = res.fetchone()[0] if res else 0
                result.total += total_count

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: vector, step: 1-count, elapsed time: {elapsed_time:.3f} seconds,"
                    f" vector column: '{vector_column_name}',"
                    f" condition: '{condition}',"
                    f" got count: {total_count}"
                )

                if total_count == 0:
                    continue

                fields_expr = ", ".join([f for f in output_fields if f != "_score"])
                if "_score" not in output_fields:
                    output_fields.append("_score")
                vector_sql = (
                    f"SELECT {fields_expr}, (1 - {vector_distance_expr}) AS _score"
                    f"  FROM {index_name}"
                    f"  WHERE {filters_expr} AND {vector_column_name} IS NOT NULL AND (1 - {vector_distance_expr}) > {vector_similarity_threshold}"
                    f"  ORDER BY {vector_distance_expr}"
                    f"  APPROXIMATE LIMIT {limit if limit != 0 else vector_topn}"
                )
                if offset != 0:
                    vector_sql += f" OFFSET {offset}"
                logger.debug("OBConnection.search with vector sql: %s", vector_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(vector_sql)
                rows = res.fetchall()

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: vector, step: 2-query, elapsed time: {elapsed_time:.3f} seconds,"
                    f" select fields: '{output_fields}',"
                    f" vector column: '{vector_column_name}',"
                    f" condition: '{condition}',"
                    f" return rows count: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))
            elif fulltext_query:
                # fulltext search, usually used to search chunks in one dataset
                count_sql = f"SELECT COUNT(id) FROM {index_name} WHERE {filters_expr} AND {fulltext_match_filter}"
                logger.debug("OBConnection.search with fulltext count sql: %s", count_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(count_sql)
                total_count = res.fetchone()[0] if res else 0
                result.total += total_count

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: fulltext, step: 1-count, elapsed time: {elapsed_time:.3f} seconds,"
                    f" query text: '{fulltext_query}',"
                    f" condition: '{condition}',"
                    f" got count: {total_count}"
                )

                if total_count == 0:
                    continue

                fields_expr = ", ".join([f for f in output_fields if f != "_score"])
                if "_score" not in output_fields:
                    output_fields.append("_score")
                fulltext_sql = (
                    f"SELECT {fields_expr}, {fulltext_match_score_expr} AS _score"
                    f"  FROM {index_name}"
                    f"  WHERE {filters_expr} AND {fulltext_match_filter}"
                    f"  LIMIT {offset}, {limit if limit != 0 else fulltext_topn}"
                )
                logger.debug("OBConnection.search with fulltext sql: %s", fulltext_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(fulltext_sql)
                rows = res.fetchall()

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: fulltext, step: 2-query, elapsed time: {elapsed_time:.3f} seconds,"
                    f" select fields: '{output_fields}',"
                    f" query text: '{fulltext_query}',"
                    f" condition: '{condition}',"
                    f" return rows count: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))
            elif len(aggFields) > 0:
                # aggregation search
                assert len(aggFields) == 1, "Only one aggregation field is supported in OceanBase."
                agg_field = aggFields[0]
                if agg_field in array_columns:
                    res = self.client.perform_raw_text_sql(
                        f"SELECT {agg_field} FROM {index_name}"
                        f" WHERE {agg_field} IS NOT NULL AND {filters_expr}"
                    )
                    counts = {}
                    for row in res:
                        if row[0]:
                            if isinstance(row[0], str):
                                try:
                                    arr = json.loads(row[0])
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON array: {row[0]}")
                                    continue
                            else:
                                arr = row[0]

                            if isinstance(arr, list):
                                for v in arr:
                                    if isinstance(v, str) and v.strip():
                                        counts[v] = counts.get(v, 0) + 1

                    for v, count in counts.items():
                        result.chunks.append({
                            "value": v,
                            "count": count,
                        })
                    result.total += len(counts)
                else:
                    res = self.client.perform_raw_text_sql(
                        f"SELECT {agg_field}, COUNT(*) as count FROM {index_name}"
                        f" WHERE {agg_field} IS NOT NULL AND {filters_expr}"
                        f" GROUP BY {agg_field}"
                    )
                    for row in res:
                        result.chunks.append({
                            "value": row[0],
                            "count": int(row[1]),
                        })
                        result.total += 1
            else:
                # only filter
                fields_expr = ", ".join([f for f in output_fields if f != "_score"])
                orders: list[str] = []
                if orderBy:
                    for field, order in orderBy.fields:
                        if isinstance(column_types[field], ARRAY):
                            f = field + "_sort"
                            fields_expr += f", array_to_string({field}, ',') AS {f}"
                            field = f
                        order = "ASC" if order == 0 else "DESC"
                        orders.append(f"{field} {order}")
                order_by_expr = ("ORDER BY " + ", ".join(orders)) if len(orders) > 0 else ""
                limit_expr = f"LIMIT {offset}, {limit}" if limit != 0 else ""
                count_sql = f"SELECT COUNT(id) FROM {index_name} WHERE {filters_expr}"
                logger.debug("OBConnection.search with normal count sql: %s", count_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(count_sql)
                total_count = res.fetchone()[0] if res else 0
                result.total += total_count

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: normal, step: 1-count, elapsed time: {elapsed_time:.3f} seconds,"
                    f" condition: '{condition}',"
                    f" got count: {total_count}"
                )

                if total_count == 0:
                    continue

                filter_sql = (
                    f"SELECT {fields_expr}"
                    f"  FROM {index_name}"
                    f"  WHERE {filters_expr}"
                    f"  {order_by_expr} {limit_expr}"
                )
                logger.debug("OBConnection.search with normal sql: %s", filter_sql)

                start_time = time.time()

                res = self.client.perform_raw_text_sql(filter_sql)
                rows = res.fetchall()

                elapsed_time = time.time() - start_time
                logger.info(
                    f"OBConnection.search table {index_name}, search type: normal, step: 2-query, elapsed time: {elapsed_time:.3f} seconds,"
                    f" select fields: '{output_fields}',"
                    f" condition: '{condition}',"
                    f" return rows count: {len(rows)}"
                )

                for row in rows:
                    result.chunks.append(self._row_to_entity(row, output_fields))
        return result

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        if not self.client.check_table_exists(indexName):
            return None

        res = self.client.get(
            table_name=indexName,
            ids=[chunkId],
        )
        row = res.fetchone()
        if row is None:
            raise Exception(f"ChunkId {chunkId} not found in index {indexName}.")

        return self._row_to_entity(row, fields=list(res.keys()))

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        docs: list[dict] = []
        ids: list[str] = []
        for document in documents:
            d: dict = {}
            for k, v in document.items():
                if vector_column_pattern.match(k):
                    d[k] = v
                    continue
                if k not in column_names:
                    if "extra" not in d:
                        d["extra"] = {}
                    d["extra"][k] = v
                    continue
                if v is None:
                    d[k] = get_default_value(k)
                    continue

                if k == "kb_id" and isinstance(v, list):
                    d[k] = v[0]
                elif k == "content_with_weight" and isinstance(v, dict):
                    d[k] = json.dumps(v, ensure_ascii=False)
                elif k == "position_int":
                    d[k] = json.dumps([list(vv) for vv in v], ensure_ascii=False)
                elif isinstance(v, list):
                    # remove characters like '\t' for JSON dump
                    v = [vv.strip() for vv in v if isinstance(vv, str)]
                    d[k] = json.dumps(v, ensure_ascii=False)
                else:
                    d[k] = v

            ids.append(d["id"])
            # this is to fix https://github.com/sqlalchemy/sqlalchemy/issues/9703
            for column_name in column_names:
                if column_name not in d:
                    d[column_name] = get_default_value(column_name)
            docs.append(d)

        logger.debug("OBConnection.insert chunks: %s", docs)

        res = []
        try:
            self.client.upsert(indexName, docs)
        except Exception as e:
            logger.error(f"OBConnection.insert error: {str(e)}")
            res.append(str(e))
        return res

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        if not self.client.check_table_exists(indexName):
            return True

        condition["kb_id"] = knowledgebaseId
        filters = get_filters(condition)
        set_values: list[str] = []
        for k, v in newValue.items():
            if k == "remove":
                if isinstance(v, str):
                    set_values.append(f"{v} = NULL")
                else:
                    assert isinstance(v, dict), f"Expected str or dict for 'remove', got {type(newValue[k])}."
                    for kk, vv in v.items():
                        assert kk in array_columns, f"Column '{kk}' is not an array column."
                        set_values.append(f"{kk} = array_remove({kk}, {get_value_str(vv)})")
            elif k == "add":
                assert isinstance(v, dict), f"Expected str or dict for 'add', got {type(newValue[k])}."
                for kk, vv in v.items():
                    assert kk in array_columns, f"Column '{kk}' is not an array column."
                    set_values.append(f"{kk} = array_append({kk}, {get_value_str(vv)})")
            else:
                set_values.append(f"{k} = {get_value_str(v)}")

        update_sql = (
            f"UPDATE {indexName}"
            f" SET {', '.join(set_values)}"
            f" WHERE {' AND '.join(filters)}"
        )
        logger.debug("OBConnection.update sql: %s", update_sql)

        try:
            self.client.perform_raw_text_sql(update_sql)
            return True
        except Exception as e:
            logger.error(f"OBConnection.update error: {str(e)}")
        return False

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        if not self.client.check_table_exists(indexName):
            return 0

        condition["kb_id"] = knowledgebaseId
        try:
            res = self.client.get(
                table_name=indexName,
                ids=None,
                where_clause=[text(f) for f in get_filters(condition)],
                output_column_name=["id"],
            )
            rows = res.fetchall()
            if len(rows) == 0:
                return 0
            ids = [row[0] for row in rows]
            logger.debug(f"OBConnection.delete chunks, filters: {condition}, ids: {ids}")
            self.client.delete(
                table_name=indexName,
                ids=ids,
            )
            return len(ids)
        except Exception as e:
            logger.error(f"OBConnection.delete error: {str(e)}")
        return 0

    @staticmethod
    def _row_to_entity(data: Row, fields: list[str]) -> dict:
        entity = {}
        for i, field in enumerate(fields):
            value = data[i]
            if value is None:
                continue
            entity[field] = get_column_value(field, value)
        return entity

    """
    Helper functions for search result
    """

    def getTotal(self, res) -> int:
        return res.total

    def getChunkIds(self, res) -> list[str]:
        return [row["id"] for row in res.chunks]

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        result = {}
        for row in res.chunks:
            data = {}
            for field in fields:
                v = row.get(field)
                if v is not None:
                    data[field] = v
            result[row["id"]] = data
        return result

    def highlight(self, txt: str, keywords: list[str]) -> str:
        boundary_chars = set(" .?/'\"()!,:;-。？！，；：""''（）【】《》……\n\t")

        result = txt
        found = False

        for keyword in keywords:
            if not keyword:
                continue

            new_result = ""
            i = 0

            while i < len(result):
                pos = result.lower().find(keyword.lower(), i)
                if pos == -1:
                    new_result += result[i:]
                    break

                new_result += result[i:pos]
                actual_keyword = result[pos:pos + len(keyword)]
                prev_char = result[pos - 1] if pos > 0 else ""
                next_char = result[pos + len(keyword)] if pos + len(keyword) < len(result) else ""

                is_boundary_before = prev_char in boundary_chars or pos == 0
                is_boundary_after = next_char in boundary_chars or pos + len(keyword) == len(result)
                is_chinese_before = pos > 0 and '\u4e00' <= prev_char <= '\u9fff'
                is_chinese_after = pos + len(keyword) < len(result) and '\u4e00' <= next_char <= '\u9fff'

                if (is_boundary_before or is_chinese_before) and (is_boundary_after or is_chinese_after):
                    new_result += f"<em>{actual_keyword}</em>"
                    found = True
                else:
                    new_result += actual_keyword

                i = pos + len(keyword)

            result = new_result

        return result if found else ""

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        ans = {}
        if len(res.chunks) == 0 or len(keywords) == 0:
            return ans

        sorted_keywords = sorted(keywords, key=len, reverse=True)

        for d in res.chunks:
            txt = d.get(fieldnm)
            if not txt:
                continue

            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            for t in re.split(r"[.?!;\n]", txt):
                highlight = self.highlight(t, sorted_keywords)
                if highlight:
                    txts.append(highlight)
            if txts:
                ans[d["id"]] = "...".join(txts)
        return ans

    def getAggregation(self, res, fieldnm: str):
        if len(res.chunks) == 0:
            return []

        counts = {}
        result = []
        for d in res.chunks:
            if "value" in d and "count" in d:
                # directly use the aggregation result
                result.append((d["value"], d["count"]))
            elif fieldnm in d:
                # aggregate the values of specific field
                v = d[fieldnm]
                if isinstance(v, list):
                    for vv in v:
                        if isinstance(vv, str) and vv.strip():
                            counts[vv] = counts.get(vv, 0) + 1
                elif isinstance(v, str) and v.strip():
                    counts[v] = counts.get(v, 0) + 1

        if len(counts) > 0:
            for k, v in counts.items():
                result.append((k, v))

        return result

    """
    SQL
    """

    def sql(sql: str, fetch_size: int, format: str):
        # TODO: execute the sql generated by text-to-sql
        return None
