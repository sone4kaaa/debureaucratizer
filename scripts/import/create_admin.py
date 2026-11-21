"""Создание администратора (идемпотентно).

Использование (из корня проекта):
  python scripts/import/create_admin.py admin@example.com --username admin --password 'Secret123'
  python scripts/import/create_admin.py admin@example.com   # пароль спросят интерактивно

Если пользователь с таким email уже есть — просто повышает его до admin.
Пароль валидируется по тем же правилам, что и при регистрации через API.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


async def create_admin(email: str, username: str, password: str) -> None:
    from pydantic import ValidationError
    from sqlalchemy import select

    from backend.core.models import RegisterRequest
    from backend.db.models import Base, User
    from backend.db.session import AsyncSessionLocal, engine
    from backend.utils.security import hash_password

    # та же валидация, что в API (email, сложность пароля)
    try:
        RegisterRequest(email=email, username=username, password=password)
    except ValidationError as exc:
        print(f"Ошибка валидации: {exc}")
        sys.exit(1)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        user = (
            await session.execute(select(User).where(User.email == email))
        ).scalar_one_or_none()
        if user is not None:
            if user.role == "admin":
                print(f"[admin] {email} уже admin — ничего не меняем")
                return
            user.role = "admin"
            await session.commit()
            print(f"[admin] существующий пользователь {email} повышен до admin")
            return

        session.add(
            User(
                email=email,
                username=username,
                hashed_password=hash_password(password),
                role="admin",
            )
        )
        await session.commit()
        print(f"[admin] создан администратор {email} (username={username})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("email", help="email администратора")
    parser.add_argument("--username", default=None, help="username (по умолчанию — часть email)")
    parser.add_argument("--password", default=None, help="пароль (иначе спросит интерактивно)")
    args = parser.parse_args()

    password = args.password or getpass.getpass("Пароль администратора: ")
    username = args.username or args.email.split("@")[0]
    asyncio.run(create_admin(args.email, username, password))


if __name__ == "__main__":
    main()
