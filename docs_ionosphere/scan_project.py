import os

def scan_directory(directory):
    file_structure = {}

    for root, dirs, files in os.walk(directory):
        path = root.split(os.sep)
        subdir = '/'.join(path)
        file_structure[subdir] = files

    return file_structure

def main():
    project_dir = "/mnt/volume_externe/my_project/ionosphere"  # Remplacez par le chemin de votre projet
    structure = scan_directory(project_dir)

    with open("file_structure.txt", "w") as f:
        for key, value in structure.items():
            f.write(f"{key}:\n")
            for file in value:
                f.write(f"    {file}\n")
            f.write("\n")

    print("La structure du fichier a été enregistrée dans file_structure.txt")

if __name__ == "__main__":
    main()
