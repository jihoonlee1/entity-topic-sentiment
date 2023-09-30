import database
import re

with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT id, content FROM sentences")
	for sent_id, content in cur.fetchall():
		content = re.sub(r"\n+", " ", content)
		content = content.strip()
		cur.execute("UPDATE sentences SET content = ? WHERE id = ?", (content, sent_id))
	con.commit()
