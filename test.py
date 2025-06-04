from font_list import FONT_LIST
import requests
import time
from concurrent.futures import ThreadPoolExecutor


def check_font_url(font_name):
    url = f"https://www.fontpalace.com/font-download/{font_name}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Check if the response contains indicators of a real font page
            content = response.text.lower()
            
            # Look for common 404 indicators
            not_found_indicators = [
                "page not found",
                "404",
                "410",
                "not found",
                "error 404",
                "page does not exist",
                "sorry, we couldn't find",
                "oops! page not found"
            ]
            
            # Check if any 404 indicators are present
            for indicator in not_found_indicators:
                if indicator in content:
                    return font_name, False
            
            return font_name, True
        else:
            return font_name, False
    except Exception as e:
        return font_name, False

def main():
    results = {
        "accessible": [],
        "inaccessible": []
    }
    
    total = len(FONT_LIST)
    print(f"Checking {total} fonts...")
    
    # Use ThreadPoolExecutor to check URLs in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_font_url, font) for font in FONT_LIST]
        
        # Process results as they complete
        for i, future in enumerate(futures):
            font, accessible = future.result()
            if accessible:
                results["accessible"].append(font)
            else:
                results["inaccessible"].append(font)
            
            # Print progress
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"Progress: {i + 1}/{total}")
            
            # Add a short delay to avoid overwhelming the server
            time.sleep(0.1)
    
    # Print stats
    print("\nResults:")
    print(f"Total fonts: {total}")
    print(f"Accessible: {len(results['accessible'])} ({len(results['accessible'])/total*100:.2f}%)")
    print(f"Inaccessible: {len(results['inaccessible'])} ({len(results['inaccessible'])/total*100:.2f}%)")
    
    # Print lists
    # print("\nAccessible fonts:")
    # for font in results["accessible"]:
    #     print(f"- {font}")
    
    print(f"\nInaccessible fonts: {len(results['inaccessible'])} fonts")
    for font in results["inaccessible"]:
        print(f"- {font}")

if __name__ == "__main__":
    main()

    # accessible_fonts = set(FONT_LIST) - set(inaccessible_fonts)
    # print("[")
    # for font in accessible_fonts:
    #     print(f"\"{font}\",")
    # print("]")