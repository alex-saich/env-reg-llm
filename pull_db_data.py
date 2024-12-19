from store_pdfs_pg import get_db_connection

def get_db_connection(connection_type='local'):
    if connection_type=='local':
        return psycopg2.connect(
            host=os.getenv('DB_HOST'),
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            sslmode='require'
        )
    elif connection_type=='streamlit':
            host=st.secrets["postgres"]["host"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            port=st.secrets["postgres"]["port"]

def pull_project_names():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT project_name FROM projects")
    project_names = [row[0] for row in cur.fetchall()]
    conn.close()
    return project_names

