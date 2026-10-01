import os
import re
import sqlite3
from pathlib import Path
import xxhash
from multiprocessing import Process, Queue

from . import query_module


def save(fname, d):
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(d, f)


def clean_query(sql_query):
    if isinstance(sql_query, list):
        sql_query = sql_query[0] if sql_query else ''
    sql_query = str(sql_query)
    # 1. Try to extract SQL from a ```sql ... ``` block (or plain ``` ... ```)
    code_block = re.search(r'```(?:sql)?\n?(.*?)```', sql_query, re.DOTALL)
    if code_block:
        sql_query = code_block.group(1).strip()
    # 2. Strip any non-SQL content that appears before the first SELECT
    idx = sql_query.upper().find('SELECT')
    if idx >= 0:
        sql_query = sql_query[idx:]
    # 3. Clean formatting artifacts
    sql_query = sql_query.replace("```sql", '')
    sql_query = sql_query.replace("```", '')
    sql_query = sql_query.replace('"""', '')
    sql_query = sql_query.replace(';', '')
    # 4. Ensure the query starts with SELECT
    if 'SELECT' not in sql_query.upper()[:10]:
        sql_query = 'SELECT ' + sql_query
    return sql_query.strip()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception:
        return len(string.split())


def resolve_db_path(database, source='auto', db_file=None):
    source = (source or 'auto').lower()
    candidates = []

    # Public BIRD entrypoints resolve an absolute SQLite path from --db-root.
    # Keep it ahead of repository-relative fallbacks so the packaged runtime
    # remains portable without a copied ./dataset/BIRD tree.
    if source in ('auto', 'bird') and db_file:
        candidates.append(Path(str(db_file)))

    if source in ('auto', 'kaggle'):
        candidates.append(Path(f'./dataset/KaggleDBQA/databases/{database}/{database}.sqlite'))
    if source in ('auto', 'bird'):
        candidates.append(Path(f'./dataset/BIRD/dev_databases/{database}/{database}.sqlite'))
    if source in ('auto', 'clambsql'):
        clambsql_root = Path('./dataset/CLAMBSQL')
        if db_file:
            rel = Path(str(db_file))
            candidates.append(clambsql_root / 'database' / rel)
            candidates.append(clambsql_root / rel)
        candidates.append(clambsql_root / 'database' / database / f'{database}.sqlite')
        candidates.append(clambsql_root / database / f'{database}.sqlite')
        database_root = clambsql_root / 'database'
        if database_root.exists():
            try:
                candidates.extend(sorted(database_root.rglob(f'{database}.sqlite')))
            except Exception:
                pass

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate)

    if source == 'clambsql' or db_file:
        raise FileNotFoundError(
            f"No sqlite DB found for CLAMBSQL database={database!r}, db_file={db_file!r}"
        )
    raise FileNotFoundError(f"No sqlite DB found for {database} (checked Kaggle/BIRD/CLAMBSQL paths)")


def generate_db_schema(database, source='auto', db_file=None):
    db_path = resolve_db_path(database, source=source, db_file=db_file)

    conn = sqlite3.connect(db_path, uri=True)
    full_schema_prompt_list = []
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table[0] == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        row = cursor.fetchone()
        if row and row[0]:
            create_prompt = row[0]
            schemas[table[0]] = create_prompt
    for k, v in schemas.items():
        full_schema_prompt_list.append(v)
    schema_prompt = "\n\n".join(full_schema_prompt_list)
    cursor.close()
    conn.close()
    return schema_prompt


def evalfunc(sql_source, sql_target, database, source='kaggle', db_file=None):
    assert source in ['kaggle', 'bird', 'clambsql']
    debug = os.getenv('RUNNER_DEBUG', '').lower() in ('1', 'true', 'yes')
    db_path = resolve_db_path(database, source=source, db_file=db_file)
    if not os.path.isfile(db_path):
        print("cannot find file", db_path)
        return False, [Exception('Missing DB')]
    timeout = float(os.getenv('SQL_EVAL_TIMEOUT_SECONDS', '12'))
    output = Queue()
    query_process = Process(target=query_module.execute_query, args=(db_path, sql_source, output))
    query_process.start()
    output_hash = ''
    try:
        if debug:
            print(f"[EVAL DEBUG] db_path={db_path}")
            try:
                print(f"[EVAL DEBUG] sql_source preview: {str(sql_source)[:400]!r}")
            except Exception:
                pass
            try:
                print(f"[EVAL DEBUG] sql_target preview: {str(sql_target)[:400]!r}")
            except Exception:
                pass
        source_results = output.get(True, timeout+5)
        query_process.join(timeout)
        if query_process.is_alive():
            query_process.terminate()
            query_process.join()
            return False, [Exception('SQL query took too much time to execute.')]
        if isinstance(source_results, Exception):
            raise source_results
        if debug:
            try:
                print(f"[EVAL DEBUG] source_results length: {len(source_results)}")
                print(f"[EVAL DEBUG] source_results sample: {str(source_results[:5])}")
            except Exception:
                pass
        output_hash = xxhash.xxh128_hexdigest(str(len(source_results)), seed=123)
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        target_results = cursor.execute(sql_target).fetchall()
        if debug:
            try:
                print(f"[EVAL DEBUG] target_results length: {len(target_results)}")
                print(f"[EVAL DEBUG] target_results sample: {str(target_results[:5])}")
            except Exception:
                pass
        cursor.close()
        connection.close()
        if len(source_results) != len(target_results):
            if debug:
                print(f"[EVAL DEBUG] length mismatch source={len(source_results)} target={len(target_results)}")
            return False, []
        if 'ORDER BY' in sql_target.upper():
            for a, b in zip(source_results, target_results):
                lhs = tuple(sorted(list(a), key=lambda x: hash(x)))
                rhs = tuple(sorted(list(b), key=lambda x: hash(x)))
                output_hash = xxhash.xxh128_hexdigest(output_hash + str(lhs), seed=123)
                if lhs != rhs:
                    if debug:
                        print(f"[EVAL DEBUG] ORDER BY mismatch lhs={lhs} rhs={rhs}")
                    return False, []
        else:
            lset, rset = set(), set()
            for a, b in zip(source_results, target_results):
                lset.add(tuple(sorted(list(a), key=lambda x: hash(x))))
                rset.add(tuple(sorted(list(b), key=lambda x: hash(x))))
            output_hash = xxhash.xxh128_hexdigest(str(lset), seed=123)
            if lset != rset:
                if debug:
                    try:
                        diff_lr = list(lset - rset)[:5]
                        diff_rl = list(rset - lset)[:5]
                        print(f"[EVAL DEBUG] set mismatch diffs lset-rset sample={diff_lr} rset-lset sample={diff_rl}")
                    except Exception:
                        pass
                return False, []
    except Exception as ex:
        print(ex)
        return False, [ex]
    return True, []


def outputHash(sql_source, database, source='kaggle', db_file=None):
    if source == 'bird':
        db_path = f'./dataset/BIRD/dev_databases/{database}/{database}.sqlite'
    elif source == 'clambsql':
        db_path = resolve_db_path(database, source=source, db_file=db_file)
    else:
        db_path = f'./databases/{database}/{database}.sqlite'
    output_hash = ''
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        source_results = cursor.execute(sql_source).fetchall()
        output_hash = xxhash.xxh128_hexdigest(str(len(source_results)), seed=123)
        if 'ORDER BY' in sql_source.upper():
            for a in source_results:
                lhs = tuple(sorted(list(a), key=lambda x: hash(x)))
                output_hash = xxhash.xxh128_hexdigest(output_hash + str(lhs), seed=123)
        else:
            lset = set()
            for a in source_results:
                lset.add(tuple(sorted(list(a), key=lambda x: hash(x))))
            output_hash = xxhash.xxh128_hexdigest(str(lset), seed=123)
    except Exception:
        return False
    finally:
        try:
            cursor.close()
            connection.close()
        except Exception:
            pass
    return output_hash


def execute(sql, database, source, db_file=None):
    assert source in ['kaggle', 'bird', 'clambsql']
    if source == 'bird':
        db_path = f'./dataset/BIRD/dev_databases/{database}/{database}.sqlite'
    elif source == 'clambsql':
        db_path = resolve_db_path(database, source=source, db_file=db_file)
    else:
        db_path = f'./databases/{database}/{database}.sqlite'
    if not os.path.isfile(db_path):
        print("cannot find file")
        return False
    results = ''
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        results = cursor.execute(sql).fetchall()
    except KeyboardInterrupt:
        try:
            cursor.close()
            connection.close()
        except Exception:
            pass
        print("KeyboardInterrupt")
        return False
    except Exception as ex:
        try:
            cursor.close()
            connection.close()
        except Exception:
            pass
        print(ex)
        return False
    finally:
        try:
            cursor.close()
            connection.close()
        except Exception:
            pass
    return results
