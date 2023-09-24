import database


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT DISTINCT article_url FROM article_topic WHERE topic_id != ?", (-1, ))
	for url, in cur.fetchall():
		cur.execute("SELECT topic_id FROM article_topic WHERE article_url = ?", (url, ))
		for topic_id, in cur.fetchall():
			if topic_id == -1:
				print("boo")

