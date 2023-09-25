import database
import re


with database.connect() as con:
	cur = con.cursor()
	topics = []
	sentiment_valid = ["positive", "negative", "neutral"]
	cur.execute("SELECT name FROM topics")
	for name, in cur.fetchall():
		topics.append(name)
	topics = "|".join(topics)
	cur.execute("SELECT * FROM temp")
	for url, entity, response in cur.fetchall():
		cur.execute("SELECT 1 FROM article_entity_topic_sentiment WHERE article_url = ? AND entity = ?", (url, entity))
		if cur.fetchone() is not None:
			continue
		response = response.lower().strip()
		temp = re.findall(r"(land use and biodiversity|energy use and greenhouse gas emissions|emissions to air|degradation and contamination \(land\)|discharges and releases \(water\)|environmental impact of products|carbon impact of products|water use|discrimination and harassment|forced labour|freedom of association|health and safety|labour relations|other labour standards|child labour|false or deceptive marketing|data privacy and security|services quality and safety|anti-competitive practices|product quality and safety|customer management|conflicts with indigenous communities|conflicts with local communities|water rights|land rights|arms export|controversial weapons|sanctions|involvement with entities violating human rights|occupied territories/disputed regions|social impact of products|media ethics|access to basic services|employees - other human rights violations|society - other human rights violations|local community - other|taxes avoidance/evasion|accounting irregularities and accounting fraud|lobbying and public policy|insider trading|bribery and corruption|remuneration|shareholder disputes/rights|board composition|corporate governance - other|intellectual property|animal welfare|resilience|business ethics - other), (positive|negative|neutral)", response)
		for row in temp:
			row = [item.strip() for item in row if item != ""]
			topic_name, sentiment = row
			cur.execute("SELECT id FROM topics WHERE name = ?", (topic_name, ))
			topic_id = cur.fetchone()
			if topic_id is not None:
				topic_id, = topic_id
				if sentiment in sentiment_valid:
					sentiment_id = sentiment_valid.index(sentiment)
					cur.execute("INSERT OR IGNORE INTO article_entity_topic_sentiment VALUES(?,?,?,?)", (url, entity, topic_id, sentiment_id))
	con.commit()
