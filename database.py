import contextlib
import sqlite3


statements = [
"""
CREATE TABLE IF NOT EXISTS topics(
	id INTEGER NOT NULL PRIMARY KEY,
	name TEXT NOT NULL,
	description TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS entities(
	id INTEGER NOT NULL PRIMARY KEY,
	name TEXT NOT NULL,
	tag TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS temp(
	topic_name TEXT NOT NULL,
	entity_name TEXT NOT NULL,
	sentiment TEXT NOT NULL,
	response TEXT NOT NULL,
	PRIMARY KEY(topic_name, entity_name, sentiment)
)
""",
"""
CREATE TABLE IF NOT EXISTS sentences(
	id INTEGER NOT NULL PRIMARY KEY,
	topic_id INTEGER NOT NULL,
	entity_id INTEGER NOT NULL,
	sentiment INTEGER NOT NULL,
	content TEXT NOT NULL
)
""",
"""
CREATE INDEX IF NOT EXISTS sentences_index0 ON sentences(entity_id, topic_id)
""",
"""
CREATE TABLE IF NOT EXISTS sentence_entity(
	sentence_id INTEGER NOT NULL,
	entity TEXT NOT NULL,
	start_pos INTEGER NOT NULL,
	end_pos INTEGER NOT NULL,
	PRIMARY KEY(sentence_id, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS temp1(
	sentence_id INTEGER NOT NULL,
	entity TEXT NOT NULL,
	start_pos INTEGER NOT NULL,
	end_pos INTEGER NOT NULL,
	response TEXT NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS irrelevant_entities(
	sentence_id INTEGER NOT NULL,
	entity TEXT NOT NULL,
	start_pos INTEGER NOT NULL,
	end_pos INTEGER NOT NULL,
	PRIMARY KEY(sentence_id, entity)
)
""",
"""
CREATE TABLE IF NOT EXISTS train(
	content TEXT NOT NULL,
	label TEXT NOT NULL
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
		with open("topics.txt") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				name, desc = line.split(":", 1)
				name = name.strip()
				desc = desc.strip()
				cur.execute("SELECT 1 FROM topics WHERE name = ?", (name, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM topics")
					topic_id, = cur.fetchone()
					cur.execute("INSERT INTO topics VALUES(?,?,?)", (topic_id, name, desc))
		with open("companies.txt") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				cur.execute("SELECT 1 FROM entities WHERE name = ?", (line, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM entities")
					entity_id, = cur.fetchone()
					cur.execute("INSERT INTO entities VALUES(?,?,?)", (entity_id, line, "ORG"))
		with open("countries.txt") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				cur.execute("SELECT 1 FROM entities WHERE name = ?", (line, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM entities")
					entity_id, = cur.fetchone()
					cur.execute("INSERT INTO entities VALUES(?,?,?)", (entity_id, line, "LOC"))
		with open("person.txt") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				cur.execute("SELECT 1 FROM entities WHERE name = ?", (line, ))
				if cur.fetchone() is None:
					cur.execute("SELECT ifnull(max(id)+1, 0) FROM entities")
					entity_id, = cur.fetchone()
					cur.execute("INSERT INTO entities VALUES(?,?,?)", (entity_id, line, "PER"))
		con.commit()


if __name__ == "__main__":
	main()
