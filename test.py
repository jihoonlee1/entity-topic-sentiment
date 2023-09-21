import database
import re


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT * FROM qa2")
	for article_url, entity, response in cur.fetchall():
		response = response.lower()
		temp = re.findall(r"\[\[(.+?)\]\]", response)
		if len(temp) == 0:
			continue
		else:
			has_topics = False
			for row in temp:
				row = row.strip()
				if row == "n/a":
					cur.execute("INSERT OR IGNORE INTO article_entity_topic VALUES(?,?,?)", (article_url, entity, -1))
					break
				else:
					topics = row.split(",")
					for topic in topics:
						topic = topic.strip()
						cur.execute("SELECT 1 FROM topics WHERE name = ?", (topic, ))
						if cur.fetchone() is not None:
							has_topics = True
							break
					if not has_topics:
						cur.execute("INSERT OR IGNORE INTO article_entity_topic VALUES(?,?,?)", (article_url, entity, -1))
					elif has_topics:
						for topic in topics:
							topic = topic.strip()
							cur.execute("SELECT id FROM topics WHERE name = ?", (topic, ))
							topic_id = cur.fetchone()
							if topic_id is not None:
								topic_id, = topic_id
								cur.execute("INSERT OR IGNORE INTO article_entity_topic VALUES(?,?,?)", (article_url, entity, topic_id))
	con.commit()
