import urllib.request
import json
import time

def test_paper_run():
    # 1. Start a paper run on Strategy 3
    data = json.dumps({"mode": "paper"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/strategies/3/runs",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    body = json.loads(res.read())
    run_id = body["data"]["id"]
    print(f"Started paper run #{run_id}")

    # 2. Wait and check status
    for i in range(10):
        time.sleep(2)
        res = urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/strategies/3/runs/{run_id}")
        status_data = json.loads(res.read())["data"]
        print(f"T+{(i+1)*2}s: status={status_data['status']}, message={status_data.get('message')}")
        if status_data["status"] == "running":
            break

    # 3. Stop the run
    req_stop = urllib.request.Request(
        f"http://127.0.0.1:8000/api/v1/strategies/3/runs/{run_id}/stop",
        data=b"",
        headers={"Content-Type": "application/json"}
    )
    res_stop = urllib.request.urlopen(req_stop)
    print(f"Stop requested: {res_stop.status}")

    # 4. Check stopped status
    for i in range(10):
        time.sleep(2)
        res = urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/strategies/3/runs/{run_id}")
        status_data = json.loads(res.read())["data"]
        print(f"T+{(i+1)*2}s after stop: status={status_data['status']}, message={status_data.get('message')}")
        if status_data["status"] in ("stopped", "completed"):
            print("Successfully verified run start and stop lifecycle!")
            break

if __name__ == "__main__":
    test_paper_run()
