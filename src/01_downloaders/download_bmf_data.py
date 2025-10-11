import requests
from pathlib import Path


def download_bmf_direct():
    """
    Download IRS EO BMF data using direct download links
    """

    # Direct download URLs for the complete dataset
    urls = [
        "https://www.irs.gov/pub/irs-soi/eo_xx.csv",  # Complete US file
        "https://www.irs.gov/pub/irs-soi/eo1.csv",  # Region 1
        "https://www.irs.gov/pub/irs-soi/eo2.csv",  # Region 2
        "https://www.irs.gov/pub/irs-soi/eo3.csv",  # Region 3
        "https://www.irs.gov/pub/irs-soi/eo4.csv",  # Region 4
    ]

    # Create the data/downloaded_irs_data directory
    download_dir = Path("data/downloaded_irs_data")
    download_dir.mkdir(parents=True, exist_ok=True)

    print("Attempting direct downloads...\n")

    for url in urls:
        filename = url.split('/')[-1]
        filepath = download_dir / filename

        try:
            print(f"Downloading {filename}...", end=" ")
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                print(f"✓ Success ({size_mb:.1f} MB)")
            else:
                print(f"✗ Not found (HTTP {response.status_code})")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\nFiles saved to: {download_dir.absolute()}")


if __name__ == "__main__":
    download_bmf_direct()