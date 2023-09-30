import database

with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT id, content FROM sentences")
	for sent_id, content, in cur.fetchall():
		if len(content) < 120:
			cur.execute("DELETE FROM sentences WHERE id = ?", (sent_id, ))
	con.commit()
