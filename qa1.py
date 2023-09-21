import database
import queue
import threading
import bingchat


def ask(input_queue, output_queue):
	while True:
		url, content = input_queue.get()
		question = f"""Sentence: "{content}"

ESG factors: [land use and biodiversity, energy use and greenhouse gas emissions, emissions to air, degradation and contamination (land), discharges and releases (water), environmental impact of products, carbon impact of products, water use, discrimination and harassment, forced labour, freedom of association, health and safety, labour relations, other labour standards, child labour, false or deceptive marketing, data privacy and security, services quality and safety, anti-competitive practices, product quality and safety, customer management, conflicts with indigenous communities, conflicts with local communities, water rights, land rights, arms export, controversial weapons, sanctions, involvement with entities violating human rights, occupied territories/disputed regions, social impact of products, media ethics, access to basic services, employees - other human rights violations, society - other human rights violations, local community - other, taxes avoidance/evasion, accounting irregularities and accounting fraud, lobbying and public policy, insider trading, bribery and corruption, remuneration, shareholder disputes/rights, board composition, corporate governance - other, intellectual property, animal welfare, resilience, business ethics - other]

Task: Based on the provided sentence and the list of ESG (Environmental, Social, and Governance) factors, please identify and list any ESG factors that are explicitly addressed in the sentence. Present the identified factors in the following format: [[factor1, factor2, ...]]. If the sentence does not explicitly address any ESG factors, indicate it as [[n/a]]."""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((url, response))
		except:
			input_queue.put((url, content))
			continue


def write(output_queue, qsize):
	remaining = qsize
	with database.connect() as con:
		cur = con.cursor()
		while True:
			url, response = output_queue.get()
			cur.execute("INSERT INTO qa1 VALUES(?,?)", (url, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{qsize}")


def main():
	with database.connect() as con:
		input_queue = queue.Queue()
		output_queue = queue.Queue()
		cur = con.cursor()
		cur.execute("SELECT article_url FROM article_is_esg WHERE is_esg = ? AND article_url NOT IN (SELECT article_url FROM qa1)", (1, ))
		for article_url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2000]
			input_queue.put((article_url, content))

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


def yep():
	import re
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT * FROM qa1 WHERE article_url NOT IN (SELECT article_url FROM qa1)")
		for url, response in cur.fetchall():
			topics = set()
			response = response.lower()
			temp = re.findall(r"\[\[(.+?)\]\]", response)
			if len(temp) == 0:
				continue
			elif len(temp) > 1:
				continue
			else:
				temp = temp[0].strip()
				if temp == "n/a":
					cur.execute("INSERT OR IGNORE INTO article_topic VALUES(?,?)", (url, -1))
				else:
					has_topics = False
					topics = temp.split(",")
					for topic in topics:
						topic = topic.strip()
						cur.execute("SELECT 1 FROM topics WHERE name = ?", (topic, ))
						if cur.fetchone() is not None:
							has_topics = True
							break
					if not has_topics:
						cur.execute("INSERT OR IGNORE INTO article_topic VALUES(?,?)", (url, -1))
					elif has_topics:
						for topic in topics:
							topic = topic.strip()
							cur.execute("SELECT id FROM topics WHERE name = ?", (topic, ))
							row = cur.fetchone()
							if row is not None:
								topic_id, = row
								cur.execute("INSERT OR IGNORE INTO article_topic VALUES(?,?)", (url, topic_id))
		con.commit()


if __name__ == "__main__":
	main()