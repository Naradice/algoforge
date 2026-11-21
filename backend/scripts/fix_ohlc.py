path = r"c:\Users\ksato\workspace\finance\algoforge\web\components\ohlc-chart.tsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("ls.setData(series.data);", "ls.setData(series.data as any);")
text = text.replace("candleSeries.setMarkers(allMarkers);", "candleSeries.setMarkers(allMarkers as any);")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated ohlc-chart.tsx series & markers casts")
