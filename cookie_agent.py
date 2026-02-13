from playwright.sync_api import sync_playwright
import time
from clients import cloudinary

def export_cookies(cookies, output_path="cookies.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        for c in cookies:
            domain = c["domain"]
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = c["path"]
            secure = "TRUE" if c["secure"] else "FALSE"
            expiry = str(int(c["expires"])) if c.get("expires") else str(int(time.time()) + 3600)
            name = c["name"]
            value = c["value"]
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
        return f    
        
def login_and_get_cookies():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # headless=True after first run
        context = browser.new_context()
        page = context.new_page()

        print("Opening Google login page...")
        page.goto("https://accounts.google.com/ServiceLogin?service=youtube")

        print("⚠️ Please log in manually (including 2FA if needed).")
        input("Press Enter here when login is complete and you can access YouTube...")

        # Visit YouTube to ensure cookies exist
        page.goto("https://www.youtube.com")
        cookies = context.cookies()
        file = export_cookies(cookies)
        cloud_response = cloudinary.uploader.upload(file, resource_type="file")
        
        print("✅ Cookies exported to cookies.txt")
        browser.close()
        return {"url":cloud_response["secure_url"]}


