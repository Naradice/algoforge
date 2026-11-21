import urllib.request
from bs4 import BeautifulSoup

url = 'https://www.uobgroup.com/personal/investments/economic/economic_market_strategies.html'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as res:
        soup = BeautifulSoup(res.read(), 'html.parser')
        print('Title:', soup.title.string if soup.title else 'No title')
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if any(k in href.lower() or k in text.lower() for k in ['fx', 'technical', 'insights', 'daily']):
                print(f'{text[:40]} -> {href}')
except Exception as e:
    print('Error:', e)
