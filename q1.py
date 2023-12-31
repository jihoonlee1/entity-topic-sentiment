import database
import queue
import threading
import bingchat
import time
import random


def write(output_queue, num_questions):
	remaining = num_questions
	with database.connect() as con:
		cur = con.cursor()
		while True:
			sentence_id, entity, start_pos, end_pos, response = output_queue.get()
			cur.execute("INSERT INTO temp1 VALUES(?,?,?,?,?)", (sentence_id, entity, start_pos, end_pos, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_questions}")


def ask(input_queue, output_queue):
	time.sleep(random.randint(0, 400))
	while True:
		sentence_id, entity, start_pos, end_pos, content, topic_name, sentiment = input_queue.get()
		temp = "positivity"
		if sentiment == "positive":
			temp = "positivity"
		elif sentiment == "negative":
			temp = "negativity"
		question = f"""
Sentence: {content}

The given sentence conveys a {sentiment} aspect related to "{topic_name}". Is the reason for this {temp} attributed to "{entity}"? If yes, say [[Yes]]. If no, say [[No]].
"""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((sentence_id, entity, start_pos, end_pos, response))
		except:
			input_queue.put((sentence_id, entity, start_pos, end_pos, content, topic_name, sentiment))
			continue


def main():
	with database.connect() as con:
		temp = []
		cur = con.cursor()
		cur.execute("SELECT * FROM sentence_entity")
		for sent_id, entity, start_pos, end_pos in cur.fetchall():
			cur.execute("SELECT topic_id, sentiment, content FROM sentences WHERE id = ?", (sent_id, ))
			topic_id, sentiment, content = cur.fetchone()
			cur.execute("SELECT name FROM topics WHERE id = ?", (topic_id, ))
			topic_name, = cur.fetchone()
			sentiment_str = "positive"
			if sentiment == 0:
				sentiment_str = "positive"
			elif sentiment == 1:
				sentiment_str = "negative"
			cur.execute("SELECT 1 FROM temp1 WHERE sentence_id = ? AND entity = ?", (sent_id, entity))
			if cur.fetchone() is None:
				temp.append((sent_id, entity, start_pos, end_pos, content, topic_name, sentiment_str))

		num_questions = len(temp)
		input_queue = queue.Queue()
		output_queue = queue.Queue()
		for sentence_id, entity, start_pos, end_pos, content, topic_name, sentiment in temp:
			input_queue.put((sentence_id, entity, start_pos, end_pos, content, topic_name, sentiment))

		ask_threads = []
		num_workers = 400
		for _ in range(num_workers):
			t = threading.Thread(target=ask, args=(input_queue, output_queue))
			t.start()
			ask_threads.append(t)

		write_thread = threading.Thread(target=write, args=(output_queue, num_questions))
		write_thread.start()

		for t in ask_threads:
			t.join()

		write_thread.join()


main()
