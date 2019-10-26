import sqlite3
import datetime
import json

DATABASE = '/home/baohoang235/WorkSpace/face-check-in/database.db'

class CheckinManager(object):
    def __init__(self, database):
        self.conn = None
        self.cursor = None
        # self.cameraNum = cameraNum

        if database:
            self.open(database)

    def open(self, database):
        try:
            self.conn = sqlite3.connect(database)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print("Error connecting to database!")

    def close(self):
        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()
    
    def create_table(self):
        self.conn.execute('\
            CREATE TABLE IF NOT EXISTS "people" (\
                "id"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\
                "name"	TEXT NOT NULL\
            )\
        ')
        self.conn.execute('\
            CREATE TABLE IF NOT EXISTS "latest" (\
                "objectID"	INTEGER NOT NULL UNIQUE,\
                "detected"	TEXT NOT NULL\
            )\
        ')
        self.conn.execute('\
            CREATE TABLE IF NOT EXISTS "checkins" (\
                "person"	INTEGER NOT NULL,\
                "time"	DATETIME NOT NULL,\
                FOREIGN KEY("person") REFERENCES "people"("id") ON DELETE CASCADE\
            )\
        ')

    def add_person(self, name):
        try:
            self.cursor.execute("INSERT INTO people (name) VALUES(?)", (name,))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.OperationalError as e:
            print(e)
            pass

    def get_name(self, uid):
        result =  self.cursor.execute("SELECT name FROM people WHERE id=?", (uid,)).fetchone()
        if result != None:
            return result[0]
        return None

    def get_id(self, name):
        result =  self.cursor.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
        if result != None:
            return result[0]
        return None
    
    def get_all_people(self):
        return self.cursor.execute("SELECT * FROM people").fetchall()

    def get_all_checkins(self, name=None):
        if name is None:
            return self.cursor.execute("SELECT * FROM checkins").fetchall()
        else:
            return self.cursor.execute("SELECT * FROM checkins WHERE person=?", (person,)).fetchall()

    def get_count(self, match=None):
        if match is None:
            query = "SELECT * FROM latest"
            self.cursor.execute(query)
        else:
            query = "SELECT * FROM latest WHERE objectID = ?"
            self.cursor.execute(query, (match, ))

        rows = self.cursor.fetchall()
        return rows

    def insert_checkin(self, person, t):
        try:
            self.cursor.execute("INSERT INTO checkins (person, time) VALUES(?, ?)", (person, t))
            # self.conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
            pass
    
    def insert_latest(self, predictions, match):
        try:
            self.cursor.execute(''' INSERT OR IGNORE INTO latest (objectID, detected) VALUES (?, ?)''', (match, json.dumps(predictions)))
            self.cursor.execute(''' UPDATE latest SET detected = ? WHERE objectID = ?''', (json.dumps(predictions), match))
            # self.conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
            pass
    
    def reset_latest(self):
        try:
            self.cursor.execute(''' DELETE FROM latest ''')
            # self.conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
            pass
    def delete_tables(self):
        self.cursor.execute('DROP TABLE checkins')
        self.cursor.execute('DROP TABLE people')
    def get_today(self):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        query = "SELECT people.id, people.name, checkins.time FROM checkins INNER JOIN people ON checkins.person == people.id AND strftime('%Y-%m-%d', checkins.time) == '{}'".format(today)
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows



if __name__ == "__main__":
    c = CheckinManager(DATABASE)
    # c.delete_tables()
    c.create_table()
    # c.add_person("Nghia")
    # c.add_person("Long")
    # print(c.get_all_people())
    # c.insert_checkin('Nghia', '1997-01-01 09-00-00')
    # print(c.get_today())
    # print(c.get_count())
    # print(c.get_all_checkins())
    c.close()
