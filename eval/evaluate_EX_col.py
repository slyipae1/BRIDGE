import sys
import sqlite3
import json
import argparse
import os
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp
import random
import sqlglot

random.seed(42)
import itertools
EVAL_METHOD = "robustEX_col" # one of "EX", "robustEX_col"


execution_results = None
evaluation_results = []
skeptical_gold_fail_ties = {}

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, required = True)
    parser.add_argument('--gold', type = str, required = True)
    parser.add_argument('--db_path', type = str, required = True)
    parser.add_argument('--mode', type = str, default = "greedy_search", choices=["greedy_search", "major_voting", "pass"])
    parser.add_argument('--output_dir', type = str, default = None,
                        help="Directory to save output files. Defaults to directory of --pred.")

    opt = parser.parse_args()

    return opt

def execute_sql(data_idx, db_file, sql):

    # normalize the SQL query by parsing and re-generating it using sqlite

    try:
        original_sql = sql
        parsed_recomposed = sqlglot.parse_one(sql, read='sqlite')
        sql = parsed_recomposed.sql(dialect="sqlite")
        if "TIME_TO_STR" in original_sql:
            print(original_sql)
            print(sql)
    except Exception as e:
        print(f"Error parsing SQL in execute_sql: {e}\nSQL: {sql}")
        raise e

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res) # make set hashable
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1

        # if len(execution_res) > 0:
        #     return data_idx, db_file, sql, execution_res, 1
        # elif len(execution_res) == 0:
        #     return data_idx, db_file, sql, execution_res, 0
    except:
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0

if EVAL_METHOD == "EX":
    def compare_sql(question_id, db_file, question, ground_truth, pred_sql) :
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        correctness = 0

        try:
            original_sql = pred_sql
            parsed_recomposed = sqlglot.parse_one(pred_sql, read='sqlite')
            pred_sql = parsed_recomposed.sql(dialect="sqlite")
            if "TIME_TO_STR" in original_sql:
                print(original_sql)
                print(pred_sql)
        except Exception as e:
            print(f"Error parsing SQL in execute_sql: {e}\nSQL: {pred_sql}")
            raise e

        try:
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(pred_sql)
            predicted_res = cursor.fetchall()

            # Handle multiple ground truth queries
            ground_truth_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            for gt_sql in ground_truth_list:
                try:
                    cursor.execute(gt_sql)
                    ground_truth_res = cursor.fetchall()
                    print('Successfully executed')
                    if set(predicted_res) == set(ground_truth_res):
                        correctness = 1
                        break  # Found a match, no need to check other ground truths
                except Exception as e:
                    print(f"Error executing ground truth SQL: {e}")
                    continue  # Try next ground truth

            conn.rollback()
        except:
            conn.rollback()
        finally:
            conn.close()
        return question_id, db_file, question, ground_truth, pred_sql, correctness, "", {}

elif EVAL_METHOD == "robustEX_col": # robustEX or robustEX_LLM
    def compare_sql(question_id, db_file, question, ground_truth, pred_sql):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        correctness = 0
        error = ""

        try:
            original_sql = pred_sql
            parsed_recomposed = sqlglot.parse_one(pred_sql, read='sqlite')
            pred_sql = parsed_recomposed.sql(dialect="sqlite")
            if "TIME_TO_STR" in original_sql:
                print(original_sql)
                print(pred_sql)
        except Exception as e:
            print(f"Error parsing SQL in execute_sql: {e}\nSQL: {pred_sql}")
            raise e

        try:
            conn.execute("BEGIN TRANSACTION;")

            # Execute predicted query
            cursor.execute(pred_sql)
            predicted_res = cursor.fetchall()

            # Handle multiple ground truth queries
            ground_truth_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            for gt_sql in ground_truth_list:
                try:
                    cursor.execute(gt_sql)
                    ground_truth_res = cursor.fetchall()

                    print('Successfully executed both queries')

                    # Section 1: Exact match
                    if set(predicted_res) == set(ground_truth_res):
                        correctness = 1
                        conn.rollback()
                        conn.close()
                        return question_id, db_file, question, ground_truth, pred_sql, correctness, "", {}

                    # Section 2: If either result is empty, try next ground truth
                    if not predicted_res or not ground_truth_res:
                        continue

                    # Get number of columns
                    pred_cols = len(predicted_res[0]) if predicted_res else 0
                    ground_cols = len(ground_truth_res[0]) if ground_truth_res else 0

                    # Section 3: same number of rows, different columns

                    # Case 1: Same number of columns, different order
                    if pred_cols == ground_cols:
                        # Try all permutations of column order
                        for perm in itertools.permutations(range(pred_cols)):
                            reordered_pred = [tuple(row[i] for i in perm) for row in predicted_res]
                            if set(reordered_pred) == set(ground_truth_res):
                                correctness = 1
                                conn.rollback()
                                conn.close()
                                return question_id, db_file, question, ground_truth, pred_sql, correctness, "", {}

                except Exception as e:
                    print(f"Error executing ground truth SQL: {e}")
                    continue  # Try next ground truth

            # Section 5: No matches found with any ground truth
            if EVAL_METHOD == "robustEX_LLM":
                # TODO: Implement LLM-based evaluation
                # For now, return 0
                correctness = 0
            else:
                correctness = 0

            conn.rollback()

        except Exception as e:
            print(f"Error executing predicted SQL for question {question_id}: {e}")
            conn.rollback()
            correctness = 0
            error = str(e)
        finally:
            conn.close()

        return question_id, db_file, question, ground_truth, pred_sql, correctness, error, {}


else: # robustEX or robustEX_LLM
    def compare_sql(question_id, db_file, question, ground_truth, pred_sql):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        correctness = 0
        error = ""

        try:
            original_sql = pred_sql
            parsed_recomposed = sqlglot.parse_one(pred_sql, read='sqlite')
            pred_sql = parsed_recomposed.sql(dialect="sqlite")
            if "TIME_TO_STR" in original_sql:
                print(original_sql)
                print(pred_sql)
        except Exception as e:
            print(f"Error parsing SQL in execute_sql: {e}\nSQL: {pred_sql}")
            raise e

        try:
            conn.execute("BEGIN TRANSACTION;")

            # Execute predicted query
            cursor.execute(pred_sql)
            predicted_res = cursor.fetchall()

            # Handle multiple ground truth queries
            ground_truth_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]

            for gt_sql in ground_truth_list:
                try:
                    cursor.execute(gt_sql)
                    ground_truth_res = cursor.fetchall()

                    print('Successfully executed both queries')

                    # Section 1: Exact match
                    if set(predicted_res) == set(ground_truth_res):
                        correctness = 1
                        conn.rollback()
                        conn.close()
                        return question_id, db_file, question, ground_truth, pred_sql, correctness, "", {}

                    # Section 2: If either result is empty, try next ground truth
                    if not predicted_res or not ground_truth_res:
                        continue

                    # Get number of columns
                    pred_cols = len(predicted_res[0]) if predicted_res else 0
                    ground_cols = len(ground_truth_res[0]) if ground_truth_res else 0

                    # Section 3: same number of rows, different columns

                    # Case 1: Same number of columns, different order
                    if pred_cols == ground_cols:
                        # Try all permutations of column order
                        for perm in itertools.permutations(range(pred_cols)):
                            reordered_pred = [tuple(row[i] for i in perm) for row in predicted_res]
                            if set(reordered_pred) == set(ground_truth_res):
                                correctness = 1
                                conn.rollback()
                                conn.close()
                                return question_id, db_file, question, ground_truth, pred_sql, correctness, "", {}

                    # Case 2: Predicted has more columns than ground truth
                    elif pred_cols > ground_cols:
                        # Try all combinations of ground_cols columns from pred_cols
                        for col_subset in itertools.combinations(range(pred_cols), ground_cols):
                            # For each subset, try all permutations
                            for perm in itertools.permutations(col_subset):
                                reordered_pred = [tuple(row[i] for i in perm) for row in predicted_res]
                                if set(reordered_pred) == set(ground_truth_res):
                                    correctness = 1
                                    conn.rollback()
                                    conn.close()
                                    return question_id, db_file, question, ground_truth, pred_sql, correctness, "superset col", {}

                except Exception as e:
                    print(f"Error executing ground truth SQL: {e}")
                    continue  # Try next ground truth

            # Section 5: No matches found with any ground truth
            if EVAL_METHOD == "robustEX_LLM":
                # TODO: Implement LLM-based evaluation
                # For now, return 0
                correctness = 0
            else:
                correctness = 0

            conn.rollback()

        except Exception as e:
            print(f"Error executing predicted SQL for question {question_id}: {e}")
            conn.rollback()
            correctness = 0
            error = str(e)
        finally:
            conn.close()

        return question_id, db_file, question, ground_truth, pred_sql, correctness, error, {}



def compare_sql_wrapper(args, timeout):
    '''Wrap execute_sql for timeout'''
    try:
        result = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = (*args, 0, "Time Out", {})
    except Exception as e:
        result = (*args, 0, str(e), {})
    return result

def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        print(f"Data index:{data_idx}\nSQL:\n{sql}\nTime Out!")
        print("-"*30)
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)

    return res

def execute_callback_evaluate_sql(result):
    '''Store the execution result in the collection'''
    question_id, db_file, question, ground_truth, pred_sql, correctness, remark, skeptical_data = result
    # evaluation_res = dict()
    # evaluation_res['question_id'] = question_id
    # evaluation_res["db_file"] = db_file
    # evaluation_res["question"] = question
    # evaluation_res["ground_truth"] = ground_truth
    # evaluation_res["pred_sql"] = pred_sql
    # evaluation_res["correctness"] = correctness
    if skeptical_data:
        skeptical_gold_fail_ties[question_id] = skeptical_data
    evaluation_results.append(
        {
            "question_id": question_id,
            "db_file": db_file,
            "question": question,
            "ground_truth": ground_truth,
            "pred_sql": pred_sql,
            "correctness": correctness,
            "remark": remark
        }
    )

    print('Done:', question_id, correctness) # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()

def execute_callback_execute_sqls(result):
    data_idx, db_file, sql, query_result, valid = result
    print('Done:', data_idx) # Print the progress

    execution_results.append(
        {
            "data_idx": data_idx,
            "db_file": db_file,
            "sql": sql,
            "query_result": query_result,
            "valid": valid
        }
    )

def evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=600):
    '''Execute the sqls in parallel'''
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_file, question, pred_sql, ground_truth in zip([x for x in range(len(db_files))], db_files, questions, pred_sqls, ground_truth_sqls):
        pool.apply_async(compare_sql_wrapper, args=((question_id, db_file, question, ground_truth, pred_sql), timeout), callback=execute_callback_evaluate_sql)
    pool.close()
    pool.join()

def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=600):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout), callback=execute_callback_execute_sqls)
    pool.close()
    pool.join()

def mark_invalid_sqls(db_files, sqls):
    global execution_results
    execution_results = []
    execute_sqls_parallel(db_files, sqls, num_cpus=20, timeout=600)
    execution_results = sorted(execution_results, key=lambda x:x['data_idx'])

    for idx, res in enumerate(execution_results):
        if res["valid"] == 0:
            sqls[idx] = "Error SQL"
    return sqls

def major_voting(db_files, pred_sqls, sampling_num, return_random_one_when_all_errors=True):
    global execution_results
    mj_pred_sqls = []
    execution_results = []
    # execute all sampled SQL queries to obtain their execution results
    execute_sqls_parallel(db_files, pred_sqls, num_cpus=20, timeout=600)
    execution_results = sorted(execution_results, key=lambda x:x['data_idx'])
    print("len(execution_results):", len(execution_results))

    # perform major voting
    for result_idx in range(0, len(execution_results), sampling_num):
        major_voting_counting = dict()
        execution_results_of_one_sample = execution_results[result_idx: result_idx + sampling_num]

        # if no predicted SQLs are valid
        if sum([res["valid"] for res in execution_results_of_one_sample]) == 0:
            if return_random_one_when_all_errors:
                mj_pred_sql = random.choice(execution_results_of_one_sample)["sql"] # select a random one to return
            else:
                mj_pred_sql = "Error SQL"
            mj_pred_sqls.append(mj_pred_sql)
            continue

        for res in execution_results_of_one_sample:
            if res["valid"] == 1: # skip invalid SQLs
                if res["query_result"] in major_voting_counting:
                    major_voting_counting[res["query_result"]]["votes"] += 1
                else:
                    major_voting_counting[res["query_result"]] = {"votes": 1, "sql": res["sql"]}

        # find the SQL with the max votes
        major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
        mj_pred_sql = major_vote["sql"]
        mj_pred_sqls.append(mj_pred_sql)

    return mj_pred_sqls

def run_eval(gold_file, pred_file, db_path, mode, save_pred_sqls, output_dir=None, num_cpus=20, timeout=600):
    global evaluation_results, skeptical_gold_fail_ties
    gold = json.load(open(gold_file))
    pred_results = json.load(open(pred_file))

    # reformat pred_results if in format {"id": pred_sql}
    if isinstance(pred_results, dict):
        reformatted_pred_results = []
        for qid, pred_sql in pred_results.items():
            if type(pred_sql) == list:
                reformatted_pred_results.append(
                    {
                        "id": int(qid),
                        "pred_sqls": pred_sql
                    }
                )
            else:
                reformatted_pred_results.append(
                    {
                        "id": int(qid),
                        "pred_sqls": [pred_sql]
                    }
            )
        pred_results = reformatted_pred_results

    # Compute bidirectional intersection of question IDs between pred and gold
    pred_id_set = set(res["id"] for res in pred_results)
    gold_id_set = set(data["question_id"] for data in gold)
    common_ids = pred_id_set & gold_id_set

    # Filter both to only common IDs, then sort by ID
    pred_results = [res for res in pred_results if res["id"] in common_ids]
    gold = [data for data in gold if data["question_id"] in common_ids]
    pred_results = sorted(pred_results, key=lambda x: x['id'])
    gold = sorted(gold, key=lambda x: x['question_id'])

    print(f"Common question_ids: {len(common_ids)} | Pred entries after intersection: {len(pred_results)} | Gold entries: {len(gold)}")

    if len(gold) == 0:
        print("WARNING: No common question_ids between pred and gold. Nothing to evaluate.")
        return 0.0, []

    # Verify ID alignment after intersection and sorting
    pred_ids = [res["id"] for res in pred_results]
    gold_ids = [data["question_id"] for data in gold]
    assert pred_ids == gold_ids, \
        f"IDs do not match pairwise after intersection/sorting. Pred IDs: {pred_ids[:5]}..., Gold IDs: {gold_ids[:5]}..."

    # Store gold question_ids in order for downstream mapping
    gold_question_ids = [d["question_id"] for d in gold]

    # Resolve output directory for saving results
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pred_file))
    os.makedirs(output_dir, exist_ok=True)
    gold_basename = os.path.splitext(os.path.basename(gold_file))[0]
    pred_basename = os.path.splitext(os.path.basename(pred_file))[0]
    output_base = os.path.join(output_dir, f"{pred_basename}_{gold_basename}")

    db_files = [os.path.join(db_path, data["db_id"], data["db_id"] + ".sqlite") for data in gold]
    questions = [data["question"] for data in gold]
    pred_sql_key = "pred_sqls"

    # Handle both single SQL string and list of SQLs for ground truth
    ground_truth_sqls = []
    for data in gold:
        sql_field = data["SQL"]
        if isinstance(sql_field, list):
            ground_truth_sqls.append(sql_field)
        else:
            ground_truth_sqls.append([sql_field])  # Convert single SQL to list

    if mode == "greedy_search":
        pred_sqls = [res[pred_sql_key][0] for res in pred_results]

        # save the (greedy-search) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(output_base + "_predsql_gd.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(pred_sqls, indent=2 ,ensure_ascii=False))

        assert len(pred_results) == len(pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(db_files, questions, pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])

        evaluation_scores = [res["correctness"] for res in evaluation_results]
        for res in evaluation_results:
            if res["correctness"] == 0:
                print("question:", res["question"])
                print("GT:", res["ground_truth"])
                print("Pred:", res["pred_sql"])
                print("-"*30)
        print("EX Accuracy (greedy search):", f"len: {sum(evaluation_scores)}/{len(evaluation_scores)}, acc: {sum(evaluation_scores)/len(evaluation_scores)}")

        # score after excluding skip_ids (optional file)
        skip_ids_path = os.path.join(os.path.dirname(gold_file), "skip_ids") # each line one question_id
        if os.path.exists(skip_ids_path):
            with open(skip_ids_path, "r", encoding="utf-8") as f:
                skip_ids = set(map(int, f.read().strip().splitlines()))
            filtered_scores = [score for idx, score in enumerate(evaluation_scores) if gold_question_ids[idx] not in skip_ids]
            if len(filtered_scores) > 0:
                print("EX Accuracy (greedy search, excluding skip_ids):", f"len: {len(filtered_scores)}, acc: {sum(filtered_scores)/len(filtered_scores)}")
            else:
                print("EX Accuracy (greedy search, excluding skip_ids): all entries skipped, no score computed")
        else:
            print(f"skip_ids file not found at {skip_ids_path}, skipping filtering")

        # save the evaluation results
        with open(output_base + f"_evalRes_gd_{EVAL_METHOD}.json", "w", encoding="utf-8") as f:
            for entry in evaluation_results:
                actual_qid = gold_question_ids[entry["question_id"]]
                if entry["question_id"] in skeptical_gold_fail_ties:
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"] = entry.copy()
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"]['question_id'] = actual_qid
                # reassign the question_id to match gold file
                entry["question_id"] = actual_qid
                for key in ["db_file", "question", "ground_truth"]: # del keys that may be too long
                    if key in entry:
                        del entry[key]
            f.write(json.dumps(evaluation_results, indent=2 ,ensure_ascii=False))
        print(f"Eval results saved to {output_base + f'_evalRes_gd_{EVAL_METHOD}.json'}")

        # save skeptical_gold_fail_ties info
        # with open(f"_skeptical_gold_fail_ties.json", "w", encoding="utf-8") as f:
        #     f.write(json.dumps(skeptical_gold_fail_ties, indent=2 ,ensure_ascii=False))

        return sum(evaluation_scores)/len(evaluation_scores), pred_sqls

    elif mode == "major_voting":
        sampling_num = len(pred_results[0][pred_sql_key])
        print("sampling_num:", sampling_num)

        db_files_expanded = []
        ground_truth_sqls_expanded = []
        for i, gold_data in enumerate(gold):
            db_files_expanded.extend([os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite")] * sampling_num)
            ground_truth_sqls_expanded.extend([ground_truth_sqls[i]] * sampling_num)

        pred_sqls = []
        for pred_data in pred_results:
            pred_sqls.extend(pred_data[pred_sql_key])
        assert len(pred_sqls) == len(db_files_expanded)

        mj_pred_sqls = major_voting(db_files_expanded, pred_sqls, sampling_num)

        # save the (major-voting) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(output_base + "_predsql_mj.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(mj_pred_sqls, indent=2 ,ensure_ascii=False))

        # reset db_files
        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        assert len(mj_pred_sqls) == len(db_files) == len(questions) == len(ground_truth_sqls)

        evaluation_results = []
        evaluate_sqls_parallel(db_files, questions, mj_pred_sqls, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]
        print("EX Accuracy (major voting):", sum(evaluation_scores)/len(evaluation_scores))

        # score after excluding skip_ids (optional file)
        skip_ids_path = os.path.join(os.path.dirname(gold_file), "skip_ids") # each line one question_id
        if os.path.exists(skip_ids_path):
            with open(skip_ids_path, "r", encoding="utf-8") as f:
                skip_ids = set(map(int, f.read().strip().splitlines()))
            filtered_scores = [score for idx, score in enumerate(evaluation_scores) if gold_question_ids[idx] not in skip_ids]
            if len(filtered_scores) > 0:
                print("EX Accuracy (major voting, excluding skip_ids):", sum(filtered_scores)/len(filtered_scores))
            else:
                print("EX Accuracy (major voting, excluding skip_ids): all entries skipped, no score computed")
        else:
            print(f"skip_ids file not found at {skip_ids_path}, skipping filtering")

        # save the evaluation results
        with open(output_base + f"_evalRes_mj_{EVAL_METHOD}.json", "w", encoding="utf-8") as f:
            for entry in evaluation_results:
                actual_qid = gold_question_ids[entry["question_id"]]
                if entry["question_id"] in skeptical_gold_fail_ties:
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"] = entry.copy()
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"]['question_id'] = actual_qid
                entry["question_id"] = actual_qid
                for key in ["db_file", "question", "ground_truth"]: # del keys that may be too long
                    if key in entry:
                        del entry[key]
            f.write(json.dumps(evaluation_results, indent=2 ,ensure_ascii=False))

        # save skeptical_gold_fail_ties info
        with open(output_base + f"_skeptical_gold_fail_ties_{EVAL_METHOD}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(skeptical_gold_fail_ties, indent=2 ,ensure_ascii=False))

        return sum(evaluation_scores)/len(evaluation_scores), mj_pred_sqls

    elif mode == "pass":
        # Get all predictions for each question (variable length lists)
        db_files = []
        for gold_data in gold:
            db_files.append(os.path.join(db_path, gold_data["db_id"], gold_data["db_id"] + ".sqlite"))

        best_scores = []
        best_pred_sqls = []  # Store the best SQL for each question

        for idx, (pred_data, db_file, question, ground_truth) in enumerate(zip(pred_results, db_files, questions, ground_truth_sqls)):
            pred_sqls_for_question = pred_data[pred_sql_key]  # Variable length list
            question_scores = []
            question_results = []

            # Evaluate all predictions for this question
            for pred_sql in pred_sqls_for_question:
                try:
                    question_id, _, _, _, _, correctness, remark, skeptical_data = compare_sql(idx, db_file, question, ground_truth, pred_sql)
                    question_scores.append(correctness)
                    question_results.append({
                        "question_id": question_id,
                        "db_file": db_file,
                        "question": question,
                        "ground_truth": ground_truth,
                        "pred_sql": pred_sql,
                        "correctness": correctness,
                        "remark": remark,
                        "skeptical_data": skeptical_data
                    })
                except Exception as e:
                    print(f"Error evaluating SQL for question {idx}: {e}")
                    question_scores.append(0)
                    question_results.append({
                        "question_id": idx,
                        "db_file": db_file,
                        "question": question,
                        "ground_truth": ground_truth,
                        "pred_sql": pred_sql,
                        "correctness": 0,
                        "remark": str(e),
                        "skeptical_data": {}
                    })

            # Find the best result for this question
            if question_scores:
                best_idx = question_scores.index(max(question_scores))
                best_result = question_results[best_idx]
                best_score = question_scores[best_idx]
                best_pred_sql = best_result["pred_sql"]

                # Handle skeptical data if present
                if best_result["skeptical_data"]:
                    skeptical_gold_fail_ties[idx] = best_result["skeptical_data"]
            else:
                best_score = 0
                best_pred_sql = "Error SQL"
                best_result = {
                    "question_id": idx,
                    "db_file": db_file,
                    "question": question,
                    "ground_truth": ground_truth,
                    "pred_sql": "Error SQL",
                    "correctness": 0,
                    "remark": "No valid predictions"
                }

            best_scores.append(best_score)
            best_pred_sqls.append(best_pred_sql)

            # Store evaluation result (excluding skeptical_data for saving)
            evaluation_results.append({
                "question_id": best_result["question_id"],
                "db_file": best_result["db_file"],
                "question": best_result["question"],
                "ground_truth": best_result["ground_truth"],
                "pred_sql": best_result["pred_sql"],
                "correctness": best_result["correctness"],
                "remark": best_result["remark"]
            })

            print(f'Done: {idx}, best score: {best_score}')

        # save the (pass mode) predicted SQL so we can check it out later
        if save_pred_sqls:
            with open(output_base + "_predsql_pass.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(best_pred_sqls, indent=2, ensure_ascii=False))

        # sort evaluation_results by question_id
        evaluation_results = sorted(evaluation_results, key=lambda x: x['question_id'])
        evaluation_scores = [res["correctness"] for res in evaluation_results]

        for res in evaluation_results:
            if res["correctness"] == 0:
                print("question:", res["question"])
                print("GT:", res["ground_truth"])
                print("Pred:", res["pred_sql"])
                print("-"*30)

        overall_score = sum(best_scores) / len(best_scores)
        print(f"EX Accuracy (pass mode):", f"len: {len(evaluation_scores)}, acc: {overall_score}")

        # overall score excluding skip_ids
        skip_ids_path = os.path.join(os.path.dirname(gold_file), "skip_ids") # each line one question_id
        if os.path.exists(skip_ids_path):
            with open(skip_ids_path, "r") as f:
                skip_ids = set([int(line.strip()) for line in f])
            filtered_scores = [score for idx, score in enumerate(best_scores) if gold_question_ids[idx] not in skip_ids]
            if len(filtered_scores) > 0:
                filtered_overall_score = sum(filtered_scores) / len(filtered_scores)
                print(f"EX Accuracy (pass mode, excluding skip_ids):", f"len: {len(filtered_scores)}, acc: {filtered_overall_score}")
            else:
                print(f"EX Accuracy (pass mode, excluding skip_ids): all entries skipped, no score computed")
        else:
            print(f"skip_ids file not found at {skip_ids_path}, skipping filtering")

        # save the evaluation results
        with open(output_base + f"_evalRes_pass_{EVAL_METHOD}.json", "w", encoding="utf-8") as f:
            for entry in evaluation_results:
                actual_qid = gold_question_ids[entry["question_id"]]
                if entry["question_id"] in skeptical_gold_fail_ties:
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"] = entry.copy()
                    skeptical_gold_fail_ties[entry["question_id"]]["evalRes"]['question_id'] = actual_qid
                # reassign the question_id to match gold file
                entry["question_id"] = actual_qid
                for key in ["db_file", "question", "ground_truth"]: # del keys that may be too long
                    if key in entry:
                        del entry[key]
            f.write(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

        # save skeptical_gold_fail_ties info
        with open(output_base + f"_skeptical_gold_fail_ties_{EVAL_METHOD}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(skeptical_gold_fail_ties, indent=2, ensure_ascii=False))

        return overall_score, best_pred_sqls


if __name__ == "__main__":
    opt = parse_option()
    run_eval(opt.gold, opt.pred, opt.db_path, opt.mode, False, output_dir=opt.output_dir)
