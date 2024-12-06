from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session

from config.config import DefaultConfig

engine = create_engine(**DefaultConfig.MYSQL_CONFIG)
Session = sessionmaker(bind=engine)
Session = scoped_session(Session)
