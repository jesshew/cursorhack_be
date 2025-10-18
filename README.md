# FastAPI Supabase Project

This is a FastAPI application configured to connect to a Supabase PostgreSQL database.

## Setup and Installation

### 1. Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

**On macOS and Linux:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

**On Windows:**
```bash
python -m venv myenv
.\myenv\Scripts\activate
```

### 2. Install Dependencies

Install all the required packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

This project uses a `.env` file to manage environment variables.

**1. Create the `.env` file:**
Create a file named `.env` in the root directory of the project.

**2. Add your Supabase credentials:**
Copy and paste the following into the `.env` file, and replace `YOUR_SUPABASE_ANON_KEY` with the "anon" key from your Supabase project's API settings.

```
SUPABASE_URL=https://sndintanfcihwmjnlybc.supabase.co
SUPABASE_KEY=YOUR_SUPABASE_ANON_KEY
```

## Running the Application

To start the FastAPI server, run the following command from the root directory of the project:

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Health Check Endpoints

The application includes the following health check endpoints:

-   **Application Status**: `http://127.0.0.1:8000/health/`
-   **Database Status**: `http://127.0.0.1:8000/health/db`
## API Endpoints

### Upload File

- **URL**: `/storage/upload`
- **Method**: `POST`
- **Description**: Uploads a file to the Supabase storage bucket named `file`. A timestamp will be prepended to the filename to ensure uniqueness.
- **Form Data**:
    - `file`: The file to upload.

- **Example using cURL**:
```