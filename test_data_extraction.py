from modules.data_extraction import extract_linkedin_profile

profile = extract_linkedin_profile(
    "https://www.linkedin.com/in/leonkatsnelson/",
    mock=True
)

print("Profile extracted:", bool(profile))
print("Number of fields:", len(profile))
print("\nProfile data:")
print(profile)