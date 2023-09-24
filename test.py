import database


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT DISTINCT article_url FROM article_entity")
	for url, in cur.fetchall():
		cur.execute("SELECT 1 FROM article_topic WHERE article_url = ? AND topic_id = ?", (url, -1))
		if cur.fetchone() is not None:
			print(url)
