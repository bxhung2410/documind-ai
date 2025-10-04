import os
from dotenv import load_dotenv
import langchain
import chromadb

print("--- BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG ---")

# 1. Kiểm tra tải biến môi trường
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if openai_key and len(openai_key) > 10:
    print("✅ Tải OPENAI_API_KEY thành công.")
else:
    print("❌ LỖI: Không tìm thấy OPENAI_API_KEY trong file .env")

# 2. Kiểm tra các thư viện đã cài
try:
    print(f"✅ Phiên bản LangChain: {langchain.__version__}")
    print(f"✅ Phiên bản ChromaDB: {chromadb.__version__}")
    print("--- KIỂM TRA HOÀN TẤT ---")
except Exception as e:
    print(f"❌ LỖI: Có vẻ một thư viện nào đó chưa được cài đặt đúng. Lỗi: {e}")