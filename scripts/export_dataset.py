import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from server.database import SessionLocal, get_account_by_username  # noqa: E402
from server.services.training_service import TrainingService  # noqa: E402

logger = logging.getLogger("export_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def export_dataset(evn_username: str, location: str = None, output_path: str | Path = None) -> Path:
    db = SessionLocal()
    try:
        ts = TrainingService()
        acc = get_account_by_username(db, evn_username)
        if not acc:
            raise ValueError(f"Không tìm thấy tài khoản {evn_username} trong DB")
        loc = location or acc.location or "Ho Chi Minh City"
        logger.info("Đang xây dựng dataset cho user %s với location %s", evn_username, loc)
        df = ts.build_dataset_from_db(db, evn_username, loc)
        if df.empty:
            raise ValueError("Dataset rỗng, không có dữ liệu để xuất")
        if output_path:
            output = Path(output_path)
        else:
            output_dir = BASE_DIR / "datasets"
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"{evn_username}_dataset.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("Đã xuất %d dòng ra %s", len(df), output)
        return output
    finally:
        db.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Export EVN dataset to CSV for experimentation (e.g., Kaggle).")
    parser.add_argument("--username", "-u", required=True, help="EVN username cần xuất dữ liệu")
    parser.add_argument("--location", "-l", help="Override location nếu muốn")
    parser.add_argument(
        "--output",
        "-o",
        help="Đường dẫn file CSV đầu ra (mặc định: datasets/<username>_dataset.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        output_file = export_dataset(args.username, location=args.location, output_path=args.output)
        print(f"Dataset exported to {output_file}")
    except Exception as exc:
        logger.error("Xuất dataset thất bại: %s", exc, exc_info=True)
        raise SystemExit(1) from exc

