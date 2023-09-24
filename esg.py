import database
import torch


def _dataset():
	with database.connect() as con:
		dataset = []
		noesg_count = 0
		cur = con.cursor()
		cur.execute("SELECT count(DISTINCT article_url) FROM article_topic WHERE topic_id != ?", (-1, ))
		max_each, = cur.fetchone()
		cur.execute("SELECT DISTINCT article_url FROM article_topic")
		for url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			cur.execute("SELECT topic_id FROM article_topic WHERE article_url = ? LIMIT 1", (url, ))
			topic_id, = cur.fetchone()
			if topic_id == -1 and noesg_count < max_each:
				dataset.append([content, 1])
				noesg_count = noesg_count + 1
			elif topic_id != -1:
				dataset.append([content, 0])
		return dataset
