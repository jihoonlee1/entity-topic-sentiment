import contextlib
import sqlite3


statements = [
"""
CREATE TABLE IF NOT EXISTS sources(
	id         INTEGER NOT NULL PRIMARY KEY,
	url        TEXT    NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS topics(
	id       INTEGER NOT NULL PRIMARY KEY,
	name     TEXT    NOT NULL,
	category TEXT    NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS articles(
	url       TEXT    NOT NULL PRIMARY KEY,
	content   TEXT    NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS question_answer0(
	article_url TEXT NOT NULL PRIMARY KEY,
	response    TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS article_is_esg(
	article_url TEXT    NOT NULL PRIMARY KEY,
	is_esg      INTEGER NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS question_answer1(
	article_url TEXT NOT NULL,
	entity      TEXT NOT NULL,
	response    TEXT NOT NULL,
	PRIMARY KEY(article_url, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS article_entity(
	article_url TEXT    NOT NULL,
	entity      TEXT    NOT NULL,
	start_pos   INTEGER NOT NULL,
	end_pos     INTEGER NOT NULL,
	PRIMARY KEY(article_url, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS article_entity_is_relevnat(
	article_url TEXT    NOT NULL,
	entity      TEXT    NOT NULL,
	is_relevant INTEGER NOT NULL,
	PRIMARY KEY(article_url, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS Sarticle_entity_topic(
	article_url INTEGER NOT NULL,
	entity      TEXT    NOT NULL,
	topic_id    INTEGER NOT NULL,
	PRIMARY KEY(article_url, entity, topic_id)
)
"""
]


def connect(database="database.sqlite", mode="rw"):
	return contextlib.closing(sqlite3.connect(f"file:{database}?mode={mode}", uri=True))


def initialize():
	with connect(mode="rwc") as con:
		cur = con.cursor()
		for st in statements:
			cur.execute(st)


def insert_sources():
	with connect() as con:
		cur = con.cursor()
		with open("sources.txt") as f:
			lines = f.readlines()
			for item in lines:
				item = item.strip()
				cur.execute("SELECT 1 FROM sources WHERE url = ?", (item, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM sources")
					source_id, = cur.fetchone()
					cur.execute("INSERT INTO sources VALUES(?,?)", (source_id, item))
			con.commit()


def insert_topics():
	with connect() as con:
		cur = con.cursor()
		with open("topics.txt") as f:
			lines = f.readlines()
			for item in lines:
				item = item.strip().lower()
				name, category = item.split(",", 1)
				name = name.strip()
				category = category.strip()
				cur.execute("SELECT 1 FROM topics WHERE name = ? AND category = ?", (name, category))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM topics")
					topic_id, = cur.fetchone()
					cur.execute("INSERT INTO topics VALUES(?,?,?)", (topic_id, name, category))
			con.commit()


if __name__ == "__main__":
	initialize()
	insert_sources()
	insert_topics()
