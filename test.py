import database
import re


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT * FROM question_answer1")
	count = 0
	for article_url, entity, response in cur.fetchall():
		irrelevant = False
		has_topics = False
		temp = re.findall(r"\[(.+?)\]", response)
		topics = set()
		if len(temp) == 0:
			continue
		for row in temp:
			row = row.strip().lower()
		topics.add(row)
		for item in topics:
			if item == "n/a":
				print("boo")