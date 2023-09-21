import database
import queue
import threading
import bingchat
import time


def ask(input_queue, output_queue):
	while True:
		url, content, topics, entity = input_queue.get()
		question = f"""Sentence: "{content}"

ESG Factors: {topics}

Identify which ESG factors were mentioned in the given sentence only because of {entity} actions or events related to {entity}. Do not include any ESG factors that were mentioned because of other entities or comparisons. Output the ESG factors that match this criterion in this format: [[factor1, factor2,…]]. If none of the ESG factors match this criterion, output [[n/a]]."""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((url, entity, response))
		except:
			input_queue.put((url, content, topics, entity))
			continue


def write(output_queue, qsize):
	remaining = qsize
	with database.connect() as con:
		cur = con.cursor()
		while True:
			url, entity, response = output_queue.get()
			cur.execute("INSERT INTO qa2 VALUES(?,?,?)", (url, entity, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{qsize}")


def main():
	with database.connect() as con:
		input_queue = queue.Queue()
		output_queue = queue.Queue()
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_topic WHERE topic_id != ?", (-1, ))
		for article_url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2000]
			topics = []
			cur.execute("SELECT topic_id FROM article_topic WHERE article_url = ?", (article_url, ))
			for topic_id, in cur.fetchall():
				cur.execute("SELECT name FROM topics WHERE id = ?", (topic_id, ))
				topic_name, = cur.fetchone()
				topics.append(topic_name)
			topics = "[" + ", ".join(topics) + "]"
			cur.execute("SELECT entity FROM article_entity WHERE article_url = ?", (article_url, ))
			for entity, in cur.fetchall():
				cur.execute("SELECT 1 FROM qa2 WHERE article_url = ? AND entity = ?", (article_url, entity))
				if cur.fetchone() is None:
					input_queue.put((article_url, content, topics, entity))

		input_threads = []
		num_workers = 400
		for _ in range(num_workers):
			t = threading.Thread(target=ask, args=(input_queue, output_queue))
			t.start()
			input_threads.append(t)

		output_thread = threading.Thread(target=write, args=(output_queue, input_queue.qsize()))
		output_thread.start()

		for t in input_threads:
			t.join()

		output_thread.join()


if __name__ == "__main__":
	main()

