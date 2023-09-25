import bingchat
import database
import threading
import queue
import re


def write(output_queue, num_questions):
	with database.connect() as con:
		remaining = num_questions
		cur = con.cursor()
		while True:
			url, entity_name, response = output_queue.get()
			cur.execute("INSERT OR IGNORE INTO temp VALUES(?,?,?)", (url, entity_name, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_questions}")


def ask(input_queue, output_queue):
	while True:
		url, content, topics, entity_name = input_queue.get()
		topics = "[" + ", ".join(topics) + "]"
		question = f"""
Sentence: "{content}"

ESG factors: {topics}

Task: Does the given sentence state the relationship between {entity_name} and any of the given ESG factors? For example, does {entity_name} have a impact on any of the given ESG factors in the given sentence? Or does any of given ESG factors have an impact on {entity_name} in the given sentence? If no, say [[n/a]]. If yes, for each relevant relationship, identify the tone of the relationship, whether it's positive, negative, or neutral. For examples, if {entity_name} is doing something good to eliminate forced labour, you would say <<forced labour, positive>>, if {entity_name} is being impacted by forced labour in a negative way, you would say <<forced labour, negative>>. Format the output this way: [[<<topic1, tone>>, <<topic2, tone>>,...]]"""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((url, entity_name, response))
		except Exception:
			input_queue.put((url, content, topics, entity_name))
			continue


def main():
	with database.connect() as con:
		temp = []
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_topic WHERE topic_id != ?", (-1, ))
		for url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			topics = []
			cur.execute("SELECT topic_id FROM article_topic WHERE article_url = ?", (url, ))
			for topic_id, in cur.fetchall():
				cur.execute("SELECT name FROM topics WHERE id = ?", (topic_id, ))
				topic_name, = cur.fetchone()
				topics.append(topic_name)
			cur.execute("SELECT entity, start_pos, end_pos FROM article_entity WHERE article_url = ?", (url, ))
			for entity_name, start_pos, end_pos in cur.fetchall():
				can_add = False
				if content[start_pos:end_pos] == entity_name:
					can_add = True
				else:
					if entity_name in content:
						can_add = True
				if can_add:
					cur.execute("SELECT 1 FROM temp WHERE article_url = ? AND entity = ?", (url, entity_name))
					if cur.fetchone() is None:
						temp.append((url, content, topics, entity_name))

	num_questions = len(temp)

	input_queue = queue.Queue()
	output_queue = queue.Queue()
	for url, content, topics, entity_name in temp:
		input_queue.put((url, content, topics, entity_name))

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


if __name__ == "__main__":
	main()
