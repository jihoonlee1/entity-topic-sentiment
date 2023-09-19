import database
import bingchat
import threading
import queue


def write(output_queue, num_articles):
	remaining = num_articles
	with database.connect() as con:
		cur = con.cursor()
		while True:
			url, response = output_queue.get()
			cur.execute("INSERT INTO question_answer0 VALUES(?,?)", (url, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_articles}")


def ask(input_queue, output_queue):
	while True:
		url, content = input_queue.get()
		try:
			conv = bingchat.conversation()
			question = f"""{content}

Does this article relate to the Environmental, Social, and Governance (ESG) factors that can impact businesses, industries, investments, and society as a whole? If yes, say [[Yes]]. If no, say [[No]]."""
			response = bingchat.ask(question, conv)
			output_queue.put((url, response))
		except:
			input_queue.put((url, content))
			continue


def main():
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT url, content FROM articles WHERE url NOT IN (SELECT article_url FROM question_answer0)")
		articles = []
		for url, content in cur.fetchall():
			content = content[:2500]
			articles.append((url, content))
		num_articles = len(articles)

	input_queue = queue.Queue()
	output_queue = queue.Queue()
	for url, content in articles:
		input_queue.put((url, content))

	num_workers = 400
	threads = []
	write_worker = threading.Thread(target=write, args=(output_queue, num_articles))
	write_worker.start()

	for _ in range(num_workers):
		t = threading.Thread(target=ask, args=(input_queue, output_queue))
		t.start()
		threads.append(t)

	for t in threads:
		t.join()

	write_worker.join()


if __name__ == "__main__":
	main()