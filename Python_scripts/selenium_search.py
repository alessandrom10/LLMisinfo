from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import psutil

def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def google_search(query, keep_browser_open=False):
    # Path to the ChromeDriver executable
    driver_path = 'C:\\Users\\flash\\Documents\\UNI\\MAG-ANNOII\\MDP\\chromedriver-win64\\chromedriver.exe'  # Sostituisci con il percorso effettivo

    # Configure Chrome options
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # Run Chrome in headless mode for faster execution
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-extensions")

    # Set up Chrome service
    service = Service(driver_path)
    
    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Open Google
        driver.get('https://www.google.com')
        
        # Wait for the search box to be present and visible
        wait = WebDriverWait(driver, 10)
        #search_box = driver.find_element(By.NAME, 'q')
        try:
            accept_cookies_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[text()="Accetta tutto"]')))
            accept_cookies_button.click()
        except Exception as e:
            print("Cookie consent dialog not found or already accepted.")
        search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
        
        # Enter the query into the search box
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for the results to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[jscontroller="SC7lYd"]')))
        
        # Extract the search results
        results = driver.find_elements(By.CSS_SELECTOR, 'div[jscontroller="SC7lYd"]')
        print("Search results found:",len(results))
        search_results = []
        
        for result in results[:5]:
            #print(result)
            try:
                title = result.find_element(By.TAG_NAME, 'h3').text
                link = result.find_element(By.TAG_NAME, 'a').get_attribute('href')
                snippet = result.find_element(By.CSS_SELECTOR, 'div[class="VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"]').text
                search_results.append({'title': title, 'link': link, 'snippet': snippet})
            except Exception as e:
                # If any element is not found, skip the result
                #continue
                search_results.append({'title': "NULL", 'link': "NULL", 'snippet': "NULL"})
        return search_results

    finally:
        if not keep_browser_open:
            # Close the browser
            driver.close()
            driver.quit()
            kill_process_and_children(service.process.pid)
        

# Example usage
query = "photosynthesis"
results = google_search(query, keep_browser_open=False)

for result in results:
    #print(f"Title: {result['title']}\nLink: {result['link']}\n")
    print(f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n")
