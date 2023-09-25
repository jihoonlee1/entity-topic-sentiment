import database
import re


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT * FROM temp")
	for url, entity, response in cur.fetchall():
		response = response.lower()
		if "[[n/a]]" in response:
			cur.execute("INSERT OR IGNORE INTO article_entity_topic_sentiment VALUES(?,?,?,?)", (url, entity, -1, -1))
	con.commit()
