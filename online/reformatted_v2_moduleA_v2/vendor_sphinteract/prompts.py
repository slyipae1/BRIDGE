# Minimal prompt constants ported from FewShotAmbSQL.ipynb

fewshot_prefix = "/* some examples are provided */\n"

feedback_prefix_v1 = "/* some examples are provided */\n"

cq_prefix_v1 = "/* some examples are provided */\n"

selfdebug_examples_prefix = '''/* Given the following incorrect sql asnwers: */
SELECT creation, COUNT(*) FROM department GROUP BY creation ORDER BY
COUNT(*) DESC LIMIT 1
/* Answer the following with no explanation: In which year were most departments established? */
SELECT creation FROM department GROUP BY creation ORDER BY COUNT(*) DESC LIMIT 1
-------
/* Given the following incorrect sql asnwers: */
SELECT customers.customer_name FROM customers JOIN orders ON customers.customer_id = orders.customer_id WHERE orders.order_status = "On Road" AND orders.order_status = "Shipped"
/* Answer the following with no explanation: Which customers have both "On Road" and "Shipped" as order status? List the customer names. */
SELECT customers.customer_name FROM customers JOIN orders ON customers.customer_id = orders.customer_id WHERE orders.order_status = "On Road" INTERSECT SELECT customers.customer_name FROM customers JOIN orders ON customers.customer_id = orders.customer_id WHERE orders.order_status = "Shipped"
-------
/* Given the following incorrect sql asnwers: */
SELECT origin FROM flight WHERE destination = "HONO"
/* Answer the following with no explanation: Show origins of all flights with destination Honolulu. */
SELECT origin FROM flight WHERE destination = "Honolulu"
-------
/* Given the following incorrect sql asnwers: */
SELECT AVG(long) FROM station WHERE id IN (SELECT station_id FROM status WHERE bikes_available <= 10)
/* Answer the following with no explanation: What is the average longitude of stations that never had bike availability more than 10? */
SELECT origin FROM flight WHERE destination = "Honolulu"
SELECT AVG(long) FROM station WHERE id NOT IN (SELECT station_id FROM status WHERE bikes_available > 10)
-------
/* Given the following incorrect sql asnwers: */
SELECT name, nationality FROM host WHERE age = (SELECT MIN(age) FROM host)
/* Answer the following with no explanation: Show the name and the nationality of the oldest host. */
SELECT name, nationality FROM host ORDER BY age DESC LIMIT 1
-------
/* Given the following incorrect sql asnwers: */
SELECT COUNT(status) FROM city
/* How many different statuses do cities have? */
SELECT COUNT(DISTINCT status) FROM city
-------'''

selfdebug_examples = selfdebug_examples_prefix.split('-------')
selfdebug_few_shot = []
for i in range(1,7):
    prefix = []
    for j in range(i):
        prefix.append(selfdebug_examples[j])
    selfdebug_few_shot.append('\n'.join(prefix))

# Prompt templates (minimal copies)
sql_generation_selfdebug = '''/* Given the following database schema: */
{schema}
/* And the following incorrect sql answers: */
{sqls}

{metadata}
/* Answer the following with no explanation: {question} */
SELECT '''

fix_invalid_v1 = """/* Given the following database schema: */
{schema}
/* And the following inexecutable sql query */
{invalidSQL}
/* And the following exception message */
{ex}

/* Fix the exception and write a new executable SQL query with no explanation */
SELECT """

sql_generation = '''/* Given the following database schema: */
{schema}

{metadata}
/* Answer the following with no explanation: {question} */
SELECT '''

sql_generation_v2 = '''/* Given the following database schema: */
{schema}
/* And the following incorrect sql answers: */
{sqls}
/* And the following user replies to help you write the correct sql query: */
{cqas}

{metadata}
/* Answer the following with no explanation: {question} */
SELECT '''

feedback_v2 = """/* Given the following Natural Language Question: */
{nlq}
/* And the following clarification question: */
{question}
/* Given the following SQL query: */
{query}

/* Answer the clarification question directly and concisely. */
"""

SRA = """/* Ask the user a new multiple choice clarification question to help you find the correct SQL answer for the following question: */
{question}
/* Given the following database schema: */
{schema}
/* And the following incorrect sql answers: */
{sqls}
/* And the following previous clarification questions and user replies: */
{cqs}

/* Consider the following ambiguity categories:
    - AmbQuestion: Is the question itself ambiguous?
    - AmbTableColumn: Is there ambiguity in mapping the entities from the QUESTION to tables and columns in the DATABASE SCHEMA?
    - AmbOutput: What fields and how many fields should be included in the output table?
    - AmbValue: What predicate value should be used to filter results?
*/

/* The clarification question should be easy to understand for people with no coding experience. */

/* Let's think step by step to generate the helpful multiple choice clarification question.
1. Summarize the clear information based on previous clarification questions and incorrect queries.
2. Evaluate whether AmbQuestion, AmbTableColumn, AmbOutput, and AmbValue remain in formulating an SQL query, considering each category individually.
3. Ask a new multiple-choice question to address the remaining ambiguities and assist in identifying the correct SQL query. Use format: mul_choice_cq = "".
*/
"""

SRA_ES = """/* Ask the user a new multiple choice clarification question to help you find the correct SQL answer for the following question: */
{question}
/* Given the following database schema: */
{schema}
/* And the following incorrect sql answers: */
{sqls}
/* And the following previous clarification questions and user replies: */
{cqs}

/* Consider the following ambiguity categories:
    - AmbQuestion: Is the question itself ambiguous?
    - AmbTableColumn: Is there ambiguity in mapping the entities from the QUESTION to tables and columns in the DATABASE SCHEMA?
    - AmbOutput: What fields and how many fields should be included in the output table?
    - AmbValue: What predicate value should be used to filter results?
*/

/* The clarification question should be easy to understand for people with no coding experience. */

/* Let's think step by step to generate the helpful multiple choice clarification question.
1. Summarize the clear information based on previous clarification questions and incorrect queries.
2. Evaluate whether AmbQuestion, AmbTableColumn, AmbOutput, and AmbValue remain in formulating an SQL query, considering each category individually.
3. If no remaining ambiguities are identified, then output "NO AMBIGUITY".
   Else, ask a new multiple-choice question to address the remaining ambiguities and assist in identifying the correct SQL query. Use format: mul_choice_cq = "".
*/
"""
