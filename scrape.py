import os
import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.webdriver import WebDriver
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # make langdetect deterministic

# ----------------------------
# CONFIG
# ----------------------------
BASE_URL = (
    "https://gepris.dfg.de/gepris/OCTOPUS?"
    "beginOfFunding=&bewilligungsStatus=&bundesland=DEU%23&context=projekt&"
    "einrichtungsart=-1&fach=%23&fachgebiet=44&fachkollegium=443&findButton=historyCall&"
    "gefoerdertIn=&ggsHunderter=0&hitsPerPage=10&index={}&nurProjekteMitAB=false&"
    "oldGgsHunderter=0&oldfachgebiet=44&pemu=%23&task=doKatalog&teilprojekte=true&"
    "zk_transferprojekt=false&language=en"
)
EDGE_DRIVER_PATH = r"C:\Users\imz\Desktop\thesis\msedgedriver.exe"
OUTPUT_CSV = "gepris_projects.csv"
SAVE_INTERVAL = 50  # save every 50 projects
MAX_PAGES = 1 # change according to number of pages per research board

# ----------------------------
# SELENIUM SETUP
# ----------------------------
def init_driver():
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")  # remove to see browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.use_chromium = True

    service = EdgeService(executable_path=EDGE_DRIVER_PATH)
    driver = webdriver.Edge(service=service, options=options)
    return driver

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def translate_text(text):
    """Detect language and translate to English if needed."""
    if not text:
        return ""
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

def extract_project_details(driver):
    """Extract title, abstract, subject area, ID, and URL, translating if needed."""
    url = driver.current_url
    m = re.search(r"/projekt/(\d+)", url)
    project_id = m.group(1) if m else None

    # Title
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        title = translate_text(title)
    except:
        title = ""

    # Abstract
    abstract = ""
    try:
        abstract = driver.find_element(By.ID, "projekttext").text.strip()
    except:
        try:
            abstract = driver.find_element(By.XPATH, "//div[contains(@class,'gepris-projektbeschreibung')]").text.strip()
        except:
            abstract = ""
    abstract = translate_text(abstract)

    # Subject Area
    subject_area = ""
    try:
        subject_area = driver.find_element(
            By.XPATH,
            "//span[@class='name' and normalize-space(text())='Subject Area']/following-sibling::span[@class='value']"
        ).text.strip()
    except:
        subject_area = ""

    return {
        "project_id": project_id,
        "title": title,
        "abstract": abstract,
        "subject_area": subject_area,
        "url": url
    }

# ----------------------------
# MAIN SCRAPER
# ----------------------------
def scrape_gepris(max_pages=MAX_PAGES):
    driver = init_driver()
    results = []
    seen_ids = set()

    # Load previously saved IDs to avoid duplicates
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        seen_ids.update(df_existing['project_id'].astype(str).tolist())
        results.extend(df_existing.to_dict(orient='records'))

    for page in range(max_pages):
        print(f"🔎 Scraping page {page + 1}/{max_pages}")
        url = BASE_URL.format(page * 50)
        driver.get(url)

        # wait for project list to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[title='Open print view']"))
            )
        except:
            print("⚠️ No items loaded on this page — skipping.")
            continue

        links = driver.find_elements(By.CSS_SELECTOR, "a[title='Open print view']")
        print(f"  → Found {len(links)} candidates")

        for link in links:
            project_url = link.get_attribute("href")
            m = re.search(r"/projekt/(\d+)", project_url)
            if not m:
                continue
            pid = m.group(1)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Open project in new tab
            driver.execute_script("window.open(arguments[0]);", project_url)
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(1)  # wait for page to load

            # Filter: only projects with Subject Area
            try:
                driver.find_element(By.XPATH, "//span[@class='name' and contains(text(),'Subject Area')]")
            except:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                continue

            # Extract details
            details = extract_project_details(driver)
            results.append(details)

            # Incremental saving
            if len(results) % SAVE_INTERVAL == 0:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"💾 Saved {len(results)} projects so far...")

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

    driver.quit()
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ DONE. {len(results)} projects saved to {OUTPUT_CSV}")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    scrape_gepris()
