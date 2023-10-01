import database
import re


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT * FROM temp")
	for topic_name, entity_name, sentiment, response in cur.fetchall():
		response = re.sub(r"\[\^([0-9])\^\]", "", response)
		response = re.sub(r"\(\^([0-9])\^\)", "", response)
		response = response.replace("**", "").strip()
		sentences = re.findall(r"\[(.+?)\]", response)
		if len(sentences) != 5:
			continue
		cur.execute("SELECT id FROM entities WHERE name = ?", (entity_name, ))
		entity_id, = cur.fetchone()
		cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name, ))
		topic_id, = cur.fetchone()
		for sent in sentences:
			sent = sent.strip()
			if entity_name not in sent:
				continue
			if len(sent) < 100:
				continue
			if not sent.endswith("."):
				sent = sent + "."
			sent = re.sub(r" +\.", ".", sent)
			sent = re.sub(r" +\,", ",", sent)
			cur.execute("SELECT ifnull(max(id)+1, 0) FROM sentences")
			sent_id, = cur.fetchone()
			sentiment_id = 0
			if sentiment == "positive":
				sentiment_id = 0
			elif sentiment == "negative":
				sentiment_id = 1
			cur.execute("INSERT INTO sentences VALUES(?,?,?,?,?)", (sent_id, topic_id, entity_id, sentiment_id, sent))
	con.commit()

