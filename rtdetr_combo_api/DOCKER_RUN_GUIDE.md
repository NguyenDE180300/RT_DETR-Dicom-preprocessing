# Docker Quick Start

1. Install Docker Engine and the Docker Compose plugin.
2. Place a compatible RT-DETR combo checkpoint at `weights/best.pt`.
3. From this directory, build and start the service:

   ```bash
   docker compose up --build
   ```

4. Open `http://127.0.0.1:8000` and upload a DICOM file.

Check service status at `http://127.0.0.1:8000/health`. To stop the service, run `docker compose down`.

The checkpoint and DICOM test files are not part of this repository. The `runs/` directory stores uploaded DICOM files and generated images; remove local files when they are no longer needed.
