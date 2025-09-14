import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import importlib  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

mod = importlib.import_module("pipeline.data.ingest_news")  # noqa: E402

SAMPLE_FEED_1 = """
<rss version="2.0">
  <channel>
    <title>Feed A</title>
    <item>
      <title>Bullish move expected</title>
      <link>https://example.com/a1</link>
      <pubDate>Wed, 11 Sep 2024 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Market down on hack</title>
      <link>https://example.com/a2</link>
      <pubDate>Wed, 11 Sep 2024 13:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
""".strip()

SAMPLE_FEED_2_DUP = """
<rss version="2.0">
  <channel>
    <title>Feed B</title>
    <item>
      <title>Duplicate link other feed</title>
      <link>https://example.com/a2</link>
      <pubDate>Wed, 11 Sep 2024 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
""".strip()


@unittest.skip("requires full environment")
class TestRSSIngest(unittest.TestCase):
    def test_rss_dedup_and_window(self):
        # monkeypatch source list
        mod._rss_default_sources = lambda: ["s1", "s2"]  # type: ignore

        async def fake_fetch_async(_sources):
            return {"s1": SAMPLE_FEED_1, "s2": SAMPLE_FEED_2_DUP}

        mod._fetch_rss_async = fake_fetch_async  # type: ignore

        start = datetime.now(timezone.utc) - timedelta(days=365)
        items, stats = mod._fetch_from_rss_window(start)
        # Expect 3 items in feed bodies, but 1 duplicate by URL -> 2 unique
        urls = {x["url"] for x in items}
        self.assertIn("https://example.com/a1", urls)
        self.assertIn("https://example.com/a2", urls)
        self.assertEqual(len(urls), 2)
        # Stats should reflect 2 sources total, both succeeded
        self.assertEqual(stats.get("rss_total"), 2.0)
        self.assertEqual(stats.get("rss_ok"), 2.0)
        self.assertEqual(stats.get("rss_fail"), 0.0)


if __name__ == "__main__":
    unittest.main()
