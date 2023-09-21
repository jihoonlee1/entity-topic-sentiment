import database
import queue
import threading
import bingchat


def ask(input_queue, output_queue):
	while True:
		url, content = input_queue.get()
		question = f"""Sentence: "{content}"

ESG Factors: [land use and biodiversity, energy use and greenhouse gas emissions, emissions to air, degradation and contamination (land), discharges and releases (water), environmental impact of products, carbon impact of products, water use, discrimination and harassment, forced labour, freedom of association, health and safety, labour relations, other labour standards, child labour, false or deceptive marketing, data privacy and security, services quality and safety, anti-competitive practices, product quality and safety, customer management, conflicts with indigenous communities, conflicts with local communities, water rights, land rights, arms export, controversial weapons, sanctions, involvement with entities violating human rights, occupied territories/disputed regions, social impact of products, media ethics, access to basic services, employees - other human rights violations, society - other human rights violations, local community - other, taxes avoidance/evasion, accounting irregularities and accounting fraud, lobbying and public policy, insider trading, bribery and corruption, remuneration, shareholder disputes/rights, board composition, corporate governance - other, intellectual property, animal welfare, resilience, business ethics - other]

Task: Does this sentence relate to the given ESG factors that can impact businesses, industries, investments, and society as a whole? If yes, say [[Yes]]. If no, say [[No]]."""
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
			cur.execute("INSERT INTO qa0 VALUES(?,?)", (url, response))
			con.commit()
			remaining = remaining - 1
			print(f"{remaining}/{qsize}")


def main():
	with database.connect() as con:
		input_queue = queue.Queue()
		output_queue = queue.Queue()
		cur = con.cursor()
		cur.execute("SELECT url, content FROM articles WHERE url NOT IN (SELECT article_url FROM qa0)")
		for url, content in cur.fetchall():
			content = content[:2000]
			input_queue.put((url, content))

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