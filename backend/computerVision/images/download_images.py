from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
from datetime import datetime
from PIL import Image
import pillow_heif  # enables HEIC support in Pillow

"""
Ausführen von ./kuehlschrankki aus
Jedes mal muss eine Authentifizierung stattfinden: kuhlschrankki Google Konto verwenden und Dienst erlauben
Programm speichert alle neu dazugekommenen Bilder basierend auf den existierenden Ordnernamen und speichert sie als jpg
"""

def get_latest_threshold(base_dir="./computerVision/images"):
    max_date = datetime(2025, 1, 1)
    for name in os.listdir(base_dir):
        if name.endswith("-download"):
            try:
                date_str = name.split("-download")[0]
                folder_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                if folder_date > max_date:
                    max_date = folder_date
            except:
                continue
    return max_date

def convert_images_to_jpg(save_folder):
    convert_errors = []
    pillow_heif.register_heif_opener()

    output_folder = os.path.join(save_folder, "converted_jpgs")

    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(save_folder):
        filepath = os.path.join(save_folder, filename)
        
        if not os.path.isfile(filepath):
            continue

        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, base_name + ".jpg")

                img.save(output_path, "JPEG", quality=90)
                print(f"✅ Converted: {filename} → {output_path}")

        except Exception as e:
            convert_errors.append(f"{filename} - {e}")
            print(f"❌ Skipped {filename} (error: {e})")
            
    return convert_errors


def main(folder_id):
    download_errors = []
    images_in_folder = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    threshold = get_latest_threshold()

    save_folder = f"computerVision/images/{datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%fZ')}-download"
    os.makedirs(save_folder, exist_ok=True)

    folder_id = "14YkFaDmL_hc_aNHhOuW1lNYa6O-B0BqqDXLEmhcctNueGSVUfhJS_3b556PtZVtZaU4lT1OJ"
    images_in_folder = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    for file in images_in_folder:
        if 'image/' not in file['mimeType']:
            print(f"Skipping non-image file: {file['title']} ({file['mimeType']})")
            continue

        title = file.get('title', '(no title)') 
        try:
            upload_time = file['sharedWithMeDate']
            upload_time = datetime.strptime(upload_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except KeyError as e:
            upload_time = file['createdDate']
            upload_time = datetime.strptime(upload_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception as e:
            download_errors.append(f"{title} - {e}")
            print(f"⚠️ Could not download '{title}': {e}")

        file_id = file['id']
        title = file['title']
        ext = "." + title.split(".")[-1] if "." in title else ".jpg"
        new_filename = os.path.join(save_folder, f"{file_id}{ext}")

        if upload_time > threshold:
            try:
                file.GetContentFile(new_filename)
                print(f"✅ Downloaded '{title}' as '{new_filename}'")
            except Exception as e:
                download_errors.append(f"{title} - {e}")
                print(f"⚠️ Could not download '{title}': {e}")
        else:
            print(f"⏭ {file['title']} ({upload_time}) is before threshold, skipping")

    convert_errors = convert_images_to_jpg(save_folder)

    errors = f"""

    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    {datetime.now()}
    Download Errors: {len(download_errors)}
    Conversion Errors: {len(convert_errors)}
    =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    """
    print(errors)
    if len(download_errors) > 0:
        print("Error Files - Downloads: ")
        print("\n".join(download_errors))
    if len(convert_errors) > 0:
        print("Error Files - Conversion: ")
        print("\n".join(convert_errors))
    
    if convert_errors or download_errors:
        with open("computerVision/images/error_log.txt", "a", encoding="utf-8") as f:
            f.write(errors)
            if download_errors:
                f.write("\nDownload Errors:\n" + "\n".join(download_errors))
            if convert_errors:
                f.write("\nConversion Errors:\n" + "\n".join(convert_errors))


if __name__ == "__main__":
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("credentials.json")
    gauth.LocalWebserverAuth(port_numbers=[8090])
    drive = GoogleDrive(gauth)

    folder_id = "14YkFaDmL_hc_aNHhOuW1lNYa6O-B0BqqDXLEmhcctNueGSVUfhJS_3b556PtZVtZaU4lT1OJ" #Unterordner Kühlschrank Ki - Bildersammlung (File Response)
    main(folder_id)
