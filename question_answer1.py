import database
import queue
import threading
import bingchat


def write(output_queue, num_articles):
	remaining = num_articles
	with database.connect() as con:
		cur = con.cursor()
		while True:
			url, entity, response = output_queue.get()
			cur.execute("INSERT INTO question_answer1 VALUES(?,?,?)", (url, entity, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_articles}")


def ask(input_queue, output_queue):
	while True:
		url, content, entity = input_queue.get()
		try:
			question = f"""Article: {content}

Topics: [land use and biodiversity, energy use and greenhouse gas emissions, emissions to air, degradation and contamination (land), discharges and releases (water), environmental impact of products, carbon impact of products, water use, discrimination and harassment, forced labour, freedom of association, health and safety, labour relations, other labour standards, child labour, false or deceptive marketing, data privacy and security, services quality and safety, anti-competitive practices, product quality and safety, customer management, conflicts with indigenous communities, conflicts with local communities, water rights, land rights, arms export, controversial weapons, sanctions, involvement with entities violating human rights, occupied territories/disputed regions, social impact of products, media ethics, access to basic services, employees - other human rights violations, society - other human rights violations, local community - other, taxes avoidance/evasion, accounting irregularities and accounting fraud, lobbying and public policy, insider trading, bribery and corruption, remuneration, shareholder disputes/rights, board composition, corporate governance - other, intellectual property, animal welfare, resilience, business ethics - other]

Is {entity} one of the main subjects in the article? If {entity} is not one of the main subjects, say [n/a]. If {entity} is one of the main subjects, identify the topics explicitly mentioned in the given article that directly impacts {entity} or directly impacted by {entity}, output the topic in this format [air emission, positive], [water rights, negative], [child labour, neutral]. If there are no topics, say [n/a]
"""
			if len(question) >= 4000:
				continue
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((url, entity, response))
		except:
			input_queue.put((url, content, entity))
			continue


def main():
	input_queue = queue.Queue()
	output_queue = queue.Queue()
	num_workers = 400

	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT article_url FROM article_is_esg WHERE is_esg = ?", (1, ))
		queries = []
		for article_url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content_max_length = 2000
			content = content[:content_max_length]
			cur.execute("SELECT entity FROM article_entity WHERE article_url = ? AND end_pos <= ?", (article_url, content_max_length))
			for entity, in cur.fetchall():
				cur.execute("SELECT 1 FROM question_answer1 WHERE article_url = ? AND entity = ?", (article_url, entity))
				if cur.fetchone() is not None:
					continue
				queries.append((article_url, content, entity))
	num_queries = len(queries)
	for article_url, content, entity in queries:
		input_queue.put((article_url, content, entity))
	write_thread = threading.Thread(target=write, args=(output_queue, num_queries))
	write_thread.start()

	threads = []
	for _ in range(num_workers):
		t = threading.Thread(target=ask, args=(input_queue, output_queue))
		t.start()
		threads.append(t)

	for t in threads:
		t.join()

	write_thread.join()


if __name__ == "__main__":
	main()