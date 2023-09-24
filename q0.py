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
			url, response = output_queue.get()
			response = re.sub(r"\n+", " ", response).strip().lower()
			response = re.findall(r"\[\[(.+?)\]\]", response)
			if len(response) != 1:
				continue
			response = response[0]
			response = response.strip()
			if response == "n/a":
				cur.execute("INSERT OR IGNORE INTO article_topic VALUES(?,?)", (url, -1))
			else:
				response = response.split(",")
				for topic_name in response:
					topic_name = topic_name.strip()
					cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name, ))
					row = cur.fetchone()
					if row is not None:
						topic_id, = row
						cur.execute("INSERT OR IGNORE INTO article_topic VALUES(?,?)", (url, topic_id))
			cur.execute("INSERT OR IGNORE INTO article_q0_done VALUES(?)", (url, ))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{num_questions}")


def ask(input_queue, output_queue):
	while True:
		url, content = input_queue.get()
		question = f"""
Sentence:
{content}

ESG Factors:
[land use and biodiversity, energy use and greenhouse gas emissions, emissions to air, degradation and contamination (land), discharges and releases (water), environmental impact of products, carbon impact of products, water use, discrimination and harassment, forced labour, freedom of association, health and safety, labour relations, other labour standards, child labour, false or deceptive marketing, data privacy and security, services quality and safety, anti-competitive practices, product quality and safety, customer management, conflicts with indigenous communities, conflicts with local communities, water rights, land rights, arms export, controversial weapons, sanctions, involvement with entities violating human rights, occupied territories/disputed regions, social impact of products, media ethics, access to basic services, employees - other human rights violations, society - other human rights violations, local community - other, taxes avoidance/evasion, accounting irregularities and accounting fraud, lobbying and public policy, insider trading, bribery and corruption, remuneration, shareholder disputes/rights, board composition, corporate governance - other, intellectual property, animal welfare, resilience, business ethics - other]

Task:
Does the given sentence explicitly state any of given ESG factors that can impact entities such as business, organization, companies, people, or location? If no, say [[n/a]]. If yes, output the chosen ESG factors from the given list in this format: [[factor1, factor2...]]
"""
		try:
			conv = bingchat.conversation()
			response = bingchat.ask(question, conv)
			output_queue.put((url, response))
		except Exception:
			input_queue.put((url, content))
			continue


def main():
	with database.connect() as con:
		temp = []
		cur = con.cursor()
		cur.execute("SELECT url, content FROM articles WHERE url NOT IN (SELECT article_url FROM article_q0_done)")
		for url, content in cur.fetchall():
			content = content[:2300]
			temp.append((url, content))
	num_questions = len(temp)

	input_queue = queue.Queue()
	output_queue = queue.Queue()
	for url, content in temp:
		input_queue.put((url, content))

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