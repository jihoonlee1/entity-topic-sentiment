import contextlib
import sqlite3


statements = [
"""
CREATE TABLE IF NOT EXISTS articles(
	url TEXT NOT NULL PRIMARY KEY,
	content TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS topics(
	id INTEGER NOT NULL PRIMARY KEY,
	name TEXT NOT NULL,
	category TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS sources(
	id INTEGER NOT NULL PRIMARY KEY,
	url TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS article_topic(
	article_url TEXT NOT NULL,
	topic_id INTEGER NOT NULL,
	PRIMARY KEY(article_url, topic_id)
)
""",
"""
CREATE TABLE IF NOT EXISTS article_entity(
	article_url TEXT NOT NULL,
	entity TEXT NOT NULL,
	start_pos INTEGER NOT NULL,
	end_pos INTEGER NOT NULL,
	PRIMARY KEY(article_url, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS article_entity_topic_sentiment(
	article_url TEXT NOT NULL,
	entity TEXT NOT NULL,
	topic_id INTEGER NOT NULL,
	sentiment INTEGER NOT NULL,
	PRIMARY KEY(article_url, entity, topic_id)
)
""",
"""
CREATE TABLE IF NOT EXISTS temp_topics(
	id INTEGER NOT NULL,
	article_url TEXT NOT NULL,
	topic_id INTEGER NOT NULL,
	PRIMARY KEY(id, article_url, topic_id)
)
"""
]

def connect(database="database.sqlite", mode="rw"):
	return contextlib.closing(sqlite3.connect(f"file:{database}?mode={mode}", uri=True))


def main():
	with connect(mode="rwc") as con:
		cur = con.cursor()
		for st in statements:
			cur.execute(st)
		with open("sources.txt") as f:
			lines = f.readlines()
			for r in lines:
				r = r.strip()
				cur.execute("SELECT 1 FROM sources WHERE url = ?", (r, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM sources")
					source_id, = cur.fetchone()
					cur.execute("INSERT INTO sources VALUES(?,?)", (source_id, r))
		with open("topics.txt") as f:
			lines = f.readlines()
			for r in lines:
				r = r.strip()
				name, category = r.split(",")
				name = name.strip()
				category = category.strip()
				cur.execute("SELECT 1 FROM topics WHERE name = ?", (name, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM topics")
					topic_id, = cur.fetchone()
					cur.execute("INSERT INTO topics VALUES(?,?,?)", (topic_id, name, category))
		con.commit()


if __name__ == "__main__":
	main()