import database
import requests
import urllib.parse
import threading
import bs4
import queue
import trafilatura
import re


def _raw_html(url):
	header = {
		"Host": urllib.parse.urlparse(url).netloc,
		"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0",
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "en-CA,en-US;q=0.7,en;q=0.3",
		"Accept-Encoding": "gzip, deflate, br",
		"Connection": "keep-alive",
		"Upgrade-Insecure-Requests": "1",
		"Sec-Fetch-Dest": "document",
		"Sec-Fetch-Mode": "navigate",
		"Sec-Fetch-Site": "same-origin",
		"Sec-Fetch-User": "?1",
		"TE": "trailers"
	}
	try:
		response = requests.get(url, headers=header)
		return response.text
	except:
		pass


def _parse_href(source_url, child_href):
	parent_parsed = urllib.parse.urlparse(source_url)
	parent_scheme = parent_parsed.scheme
	parent_netloc = parent_parsed.netloc
	parent_path = parent_parsed.path

	child_parsed = urllib.parse.urlparse(child_href)
	child_scheme = child_parsed.scheme
	child_netloc = child_parsed.netloc
	child_path = child_parsed.path

	if child_scheme != parent_scheme and not child_scheme:
		child_scheme = parent_scheme
	elif child_scheme != parent_scheme and child_scheme:
		child_scheme = None
	if child_netloc != parent_netloc and not child_netloc:
		child_netloc = parent_netloc
	elif child_netloc != parent_netloc and child_netloc:
		child_netloc = None
	if not child_path or child_path == parent_path:
		child_path = None
	if child_scheme is None or child_netloc is None or child_path is None:
		return None
	child_path = child_path + "/" if not child_path.endswith("/") else child_path
	final_path = f"{child_scheme}://{child_netloc}{child_path}".lower()
	return final_path


def scrap_page(source_id, source_url, input_queue, visited, output_queue):
	while True:
		href = input_queue.get()
		visited.add(href)
		try:
			raw_html = _raw_html(href)
			if raw_html is None:
				continue
			article_content = trafilatura.extract(raw_html)
			if article_content is not None:
				article_content = re.sub(r"\n+", " ", article_content).strip()
				if len(article_content) > 500:
					output_queue.put((href, article_content, source_id))
			soup = bs4.BeautifulSoup(raw_html, "html.parser")
			links = set()
			temp = soup.find_all("a", {"href": True})
			for item in temp:
				href = item.get("href").strip()
				href = _parse_href(source_url, href)
				if href is not None:
					links.add(href)
			for item in links:
				if item not in visited:
					visited.add(item)
					input_queue.put(item)
		except:
			continue


def open_website(source_id, source_url, output_queue):
	try:
		num_worker_per_website = 3
		input_queue = queue.Queue()
		raw_html = _raw_html(source_url)
		soup = bs4.BeautifulSoup(raw_html, "html.parser")
		print(f"Prepping: {source_url}")
		temp = soup.find_all("a", {"href": True})
		links = set()
		visited = set()
		scrap_threads = []
		for item in temp:
			href = item.get("href").strip()
			href =_parse_href(source_url, href)
			if href is not None:
				links.add(href)
		for item in links:
			input_queue.put(item)
		for _ in range(num_worker_per_website):
			t = threading.Thread(target=scrap_page, args=(source_id, source_url, input_queue, visited, output_queue))
			t.start()
			scrap_threads.append(t)

		for t in scrap_threads:
			t.join()

	except Exception as e:
		pass


def write(output_queue):
	with database.connect() as con:
		cur = con.cursor()
		while True:
			url, content, source_id = output_queue.get()
			cur.execute("SELECT 1 FROM articles WHERE url = ?", (url, ))
			if cur.fetchone() is None:
				cur.execute("INSERT INTO articles VALUES(?,?)", (url, content))
				print(f"Inserting {url}")
				con.commit()


def main():
	with database.connect() as con:
		output_queue = queue.Queue()
		write_thread = threading.Thread(target=write, args=(output_queue, ))
		cur = con.cursor()
		cur.execute("SELECT id, url FROM sources")
		scrap_threads = []
		for source_id, source_url in cur.fetchall():
			t = threading.Thread(target=open_website, args=(source_id, source_url, output_queue))
			t.start()
			scrap_threads.append(t)

		write_thread.start()

		for t in scrap_threads:
			t.join()

		write_thread.join()


if __name__ == "__main__":
	main()