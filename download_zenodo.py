import os
import requests

def get_zenodo_files(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        files = data.get('files', [])
        print(f"Found {len(files)} files.")
        for f in files:
            print(f"- {f['key']}: {f['links']['self']}")
            # Download file
            download_url = f['links']['self']
            filename = f['key']
            print(f"Downloading {filename}...")
            r = requests.get(download_url, stream=True)
            with open(os.path.join("data", filename), 'wb') as fd:
                for chunk in r.iter_content(chunk_size=8192):
                    fd.write(chunk)
            print(f"Saved {filename}")
    else:
        print(f"Failed to fetch record {record_id}. HTTP {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    get_zenodo_files("16745408")
