"""
common/dataio.py — Couche d'accès SQLite par cog.

Usage type dans un cog :

    from common.dataio import CogData, DictTableBuilder

    class MyCog(commands.Cog):
        def __init__(self, bot):
            self.data = CogData('mycog')
            self.data.set_builders(
                discord.Guild,
                DictTableBuilder('config', {'mode': 'strict', 'enabled': True}),
            )

        @commands.Cog.listener()
        async def on_guild_join(self, guild):
            s = self.data.get(guild).settings('config')
            s['mode']           # → 'strict'
            s['mode'] = 'greedy'
            s.get('enabled', cast=bool)  # → True
"""

import logging
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import discord

logger = logging.getLogger('DataIO')


# ---------------------------------------------------------------------------
# TableBuilder — définition d'une table
# ---------------------------------------------------------------------------

class TableBuilder:
    """Définit une table SQLite : requête CREATE TABLE + lignes par défaut."""

    def __init__(
        self,
        query: str,
        default_values: Sequence[dict[str, Any]] = (),
        *,
        insert_on_reconnect: bool = False,
    ) -> None:
        """
        :param query: Requête `CREATE TABLE [IF NOT EXISTS] …`
        :param default_values: Lignes insérées (INSERT OR IGNORE) à la création de la table.
        :param insert_on_reconnect: Si True, les lignes par défaut sont aussi insérées
                                    à chaque reconnexion (pratique pour les configs).
        """
        if not re.match(r'CREATE\s+TABLE', query.strip(), re.IGNORECASE):
            raise ValueError("La requête doit commencer par CREATE TABLE")
        self.query = query.strip()
        if default_values:
            keys = set(default_values[0].keys())
            if not all(set(d.keys()) == keys for d in default_values):
                raise ValueError("Toutes les valeurs par défaut doivent avoir les mêmes clés")
        self.default_values: Sequence[dict[str, Any]] = default_values
        self.insert_on_reconnect = insert_on_reconnect

    @property
    def table_name(self) -> str:
        """Nom de la table extrait de la requête."""
        m = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', self.query, re.IGNORECASE)
        if not m:
            raise ValueError(f"Nom de table introuvable dans : {self.query!r}")
        return m.group(1)

    def __repr__(self) -> str:
        return f"<TableBuilder table={self.table_name!r}>"


class DictTableBuilder(TableBuilder):
    """Raccourci pour une table clé/valeur : `(key TEXT PRIMARY KEY, value TEXT)`."""

    def __init__(
        self,
        name: str,
        defaults: dict[str, Any] = {},
        *,
        insert_on_reconnect: bool = True,
    ) -> None:
        """
        :param name: Nom de la table.
        :param defaults: Valeurs initiales `{clé: valeur}`.
        :param insert_on_reconnect: Réinsère les défauts manquants à chaque connexion.
        """
        query = f"CREATE TABLE IF NOT EXISTS {name} (key TEXT PRIMARY KEY, value TEXT)"
        rows = [{"key": k, "value": _to_str(v)} for k, v in defaults.items()]
        super().__init__(query, rows, insert_on_reconnect=insert_on_reconnect)

    def __repr__(self) -> str:
        return f"<DictTableBuilder table={self.table_name!r}>"


# ---------------------------------------------------------------------------
# Settings — accès dict-like à une table clé/valeur
# ---------------------------------------------------------------------------

class Settings:
    """Façade dict-like sur une table `(key, value)` d'une `Database`."""

    def __init__(self, db: "Database", table: str) -> None:
        self._db = db
        self._table = table

    # --- Lecture ---

    def get(self, key: str, default: Any = None, *, cast: type = str) -> Any:
        """Lit la valeur de `key`, retourne `default` si absente."""
        row = self._db.fetch(f"SELECT value FROM {self._table} WHERE key = ?", key)
        if row is None:
            return default
        return _cast(row["value"], cast)

    def all(self) -> dict[str, str]:
        """Retourne toutes les paires sous forme de `dict[str, str]`."""
        return {
            r["key"]: r["value"]
            for r in self._db.fetchall(f"SELECT key, value FROM {self._table}")
        }

    # --- Écriture ---

    def set(self, key: str, value: Any) -> None:
        """Écrit `key = value` (valeur convertie en str)."""
        self._db.execute(
            f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",
            key, _to_str(value),
        )

    def update(self, data: dict[str, Any]) -> None:
        """Écrit plusieurs paires en une seule transaction."""
        self._db.executemany(
            f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",
            [(k, _to_str(v)) for k, v in data.items()],
        )

    def delete(self, key: str) -> None:
        """Supprime la clé."""
        self._db.execute(f"DELETE FROM {self._table} WHERE key = ?", key)

    # --- Interface dict ---

    def __getitem__(self, key: str) -> str:
        row = self._db.fetch(f"SELECT value FROM {self._table} WHERE key = ?", key)
        if row is None:
            raise KeyError(key)
        return row["value"]

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete(key)

    def __contains__(self, key: str) -> bool:
        return self._db.fetch(f"SELECT 1 FROM {self._table} WHERE key = ?", key) is not None

    def __repr__(self) -> str:
        return f"<Settings table={self._table!r}>"


# ---------------------------------------------------------------------------
# Database — connexion SQLite pour un modèle
# ---------------------------------------------------------------------------

class Database:
    """Wrapper SQLite pour un modèle Discord (guild, user…) ou une base globale."""

    def __init__(self, path: Path, builders: Sequence[TableBuilder] = ()) -> None:
        self._path = path
        self.conn = self._open(path, builders)

    def _open(self, path: Path, builders: Sequence[TableBuilder]) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

        existing = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for builder in builders:
            if builder.table_name not in existing:
                conn.execute(builder.query)
                _insert_defaults(conn, builder)
                logger.debug("Table créée : %s:%s", path.name, builder.table_name)
            elif builder.insert_on_reconnect:
                _insert_defaults(conn, builder)
        conn.commit()
        return conn

    # --- SQL de base ---

    def execute(self, query: str, *args: Any, commit: bool = True) -> None:
        """Exécute une requête sans retour de données."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(query, args)
        if commit:
            self.conn.commit()

    def executemany(self, query: str, args: Iterable[Sequence[Any]], *, commit: bool = True) -> None:
        """Exécute une requête paramétrique sur plusieurs jeux de données."""
        with closing(self.conn.cursor()) as cur:
            cur.executemany(query, args)
        if commit:
            self.conn.commit()

    def fetch(self, query: str, *args: Any) -> sqlite3.Row | None:
        """Retourne la première ligne du résultat, ou None."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(query, args)
            return cur.fetchone()

    def fetchall(self, query: str, *args: Any) -> list[sqlite3.Row]:
        """Retourne toutes les lignes du résultat."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(query, args)
            return cur.fetchall()

    def commit(self) -> None:
        """Commit manuel (utile quand `commit=False` a été passé aux méthodes ci-dessus)."""
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Groupe plusieurs opérations en une seule transaction atomique.

        Exemple :
            with db.transaction():
                db.execute("INSERT …", commit=False)
                db.execute("UPDATE …", commit=False)
        """
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Ferme la connexion."""
        self.conn.close()

    # --- Raccourci settings ---

    def settings(self, table: str) -> Settings:
        """Retourne un accesseur `Settings` pour la table clé/valeur `table`."""
        return Settings(self, table)

    # --- Utilitaires ---

    @property
    def tables(self) -> list[str]:
        """Liste des tables présentes dans la base."""
        return [r[0] for r in self.fetchall("SELECT name FROM sqlite_master WHERE type='table'")]

    def __repr__(self) -> str:
        return f"<Database {self._path.name!r}>"


# ---------------------------------------------------------------------------
# CogData — point d'entrée par cog
# ---------------------------------------------------------------------------

class CogData:
    """Gestionnaire de bases de données pour un cog.

    Crée un fichier `.db` par modèle dans `cogs/<cog_name>/data/`.
    Chaque instance est indépendante ; instanciez-la dans `__init__` du cog.
    """

    def __init__(self, cog_name: str) -> None:
        self.cog_name = cog_name.lower()
        self._data_folder = Path(f"cogs/{self.cog_name}/data")
        self._data_folder.mkdir(parents=True, exist_ok=True)

        self._databases: dict[str, Database] = {}
        self._builders: dict[str, tuple[TableBuilder, ...]] = {}

    # --- Configuration des tables ---

    def set_builders(
        self,
        model_type: type[discord.abc.Snowflake] | str,
        *builders: TableBuilder,
    ) -> None:
        """Associe des `TableBuilder` à un type de modèle.

        Ils sont appliqués à chaque nouvelle connexion pour ce type.

        :param model_type: `discord.Guild`, `discord.User`, ou une clé str (ex. `'global'`).
        :param builders: Un ou plusieurs `TableBuilder` / `DictTableBuilder`.
        """
        key = _type_key(model_type)
        self._builders[key] = tuple(builders)

    # --- Accès aux bases ---

    def get(self, model: discord.abc.Snowflake | str = "global") -> Database:
        """Retourne (ou crée) la `Database` associée à ce modèle.

        :param model: Instance Discord (`guild`, `user`…) ou une clé str pour une base nommée.
        """
        cache_key = _model_cache_key(model)
        if cache_key not in self._databases:
            builders = self._builders.get(_model_type_key(model), ())
            self._databases[cache_key] = Database(
                self._data_folder / f"{cache_key}.db", builders
            )
        return self._databases[cache_key]

    # --- Fermeture / suppression ---

    def close(self, model: discord.abc.Snowflake | str) -> None:
        """Ferme la connexion de ce modèle (sans supprimer le fichier)."""
        key = _model_cache_key(model)
        if key in self._databases:
            self._databases.pop(key).close()

    def close_all(self) -> None:
        """Ferme toutes les connexions ouvertes."""
        for db in self._databases.values():
            db.close()
        self._databases.clear()

    def delete(self, model: discord.abc.Snowflake | str) -> None:
        """Ferme et supprime le fichier `.db` de ce modèle."""
        key = _model_cache_key(model)
        if key in self._databases:
            self._databases.pop(key).close()
        path = self._data_folder / f"{key}.db"
        if path.exists():
            path.unlink()

    # --- Dossiers du cog ---

    def subfolder(self, name: str, *, create: bool = False) -> Path:
        """Retourne le chemin d'un sous-dossier du cog (`cogs/<name>/<name>`)."""
        path = Path(f"cogs/{self.cog_name}") / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        return f"<CogData cog={self.cog_name!r}>"


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _to_str(value: Any) -> str:
    """Convertit une valeur en str pour stockage (bool → '0'/'1')."""
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def _cast(raw: str, cast: type) -> Any:
    """Convertit une str stockée vers le type demandé."""
    if cast is bool:
        return bool(int(raw))
    return cast(raw)


def _insert_defaults(conn: sqlite3.Connection, builder: TableBuilder) -> None:
    """Insère les lignes par défaut d'un builder (INSERT OR IGNORE)."""
    if not builder.default_values:
        return
    cols = list(builder.default_values[0].keys())
    placeholders = ", ".join("?" * len(cols))
    conn.executemany(
        f"INSERT OR IGNORE INTO {builder.table_name} ({', '.join(cols)}) VALUES ({placeholders})",
        [tuple(row[c] for c in cols) for row in builder.default_values],
    )


def _type_key(model_type: type[discord.abc.Snowflake] | str) -> str:
    if isinstance(model_type, str):
        return model_type.lower()
    return model_type.__name__.lower()


def _model_cache_key(model: discord.abc.Snowflake | str) -> str:
    if isinstance(model, discord.abc.Snowflake):
        return f"{type(model).__name__.lower()}_{model.id}"
    return re.sub(r"[^a-z0-9_]", "_", model.lower())


def _model_type_key(model: discord.abc.Snowflake | str) -> str:
    if isinstance(model, discord.abc.Snowflake):
        return type(model).__name__.lower()
    return model.lower()
