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
			topic_name, entity_name, sentiment, response = output_queue.get()
			cur.execute("INSERT INTO temp VALUES(?,?,?,?)", (topic_name, entity_name, sentiment, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_questions}")


def ask(input_queue, output_queue):
	time.sleep(random.randint(0, 400))
	while True:
		entity_name, topic_name, topic_desc, sentiment = input_queue.get()
		question = f"""Topic: {topic_name}
Topic description: {topic_desc}

Task: Write 5 sentences that describe how {entity_name} has made a {sentiment} impact on the given topic as per given topic definition. Each sentence should mention a different aspect of {entity_name}'s actions or achievements. Each sentence should include the word “{entity_name}”. Use brackets to separate each sentence. For example: [sentence1], [sentence2], [sentence3]…"""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((topic_name, entity_name, sentiment, response))
		except:
			input_queue.put((entity_name, topic_name, topic_desc, sentiment))
			continue


def main():
	with database.connect() as con:
		temp = []
		cur = con.cursor()
		cur.execute("SELECT name FROM entities")
		for entity_name, in cur.fetchall():
			cur.execute("SELECT name, description FROM topics")
			for topic_name, topic_desc in cur.fetchall():
				cur.execute("SELECT 1 FROM temp WHERE topic_name = ? AND entity_name = ? AND sentiment = ?", (topic_name, entity_name, "positive"))
				if cur.fetchone() is None:
					temp.append((entity_name, topic_name, topic_desc, "positive"))
				cur.execute("SELECT 1 FROM temp WHERE topic_name = ? AND entity_name = ? AND sentiment = ?", (topic_name, entity_name, "negative"))
				if cur.fetchone() is None:
					temp.append((entity_name, topic_name, topic_desc, "negative"))
		num_questions = len(temp)
		input_queue = queue.Queue()
		output_queue = queue.Queue()
		for entity_name, topic_name, topic_desc, sentiment in temp:
			input_queue.put((entity_name, topic_name, topic_desc, sentiment))

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
