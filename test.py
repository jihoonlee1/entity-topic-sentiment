import database
import re


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT * FROM irrelevant_entities")
	count = 0
	for sent_id, entity, start_pos, end_pos in cur.fetchall():
		cur.execute("SELECT content FROM sentences WHERE id = ?", (sent_id, ))
		content, = cur.fetchone()
		if content[start_pos:end_pos] != entity:
			print("boo")




