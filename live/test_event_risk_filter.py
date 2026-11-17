"""事件风控拦截层 - 单元测试
运行方式: cd 项目根目录 && python -m live.test_event_risk_filter
"""

import sys
import os
import tempfile
import shutil
import csv

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.event_risk_filter import EventRiskFilter


class TestEventRiskFilter:
    """单元测试集"""

    def setup_method(self):
        """每个测试前初始化，用临时目录"""
        self.tmpdir = tempfile.mkdtemp()
        self.filter = EventRiskFilter(
            keywords_path=os.path.join(self.tmpdir, "dummy.txt"),
            output_dir=self.tmpdir
        )
        # 创建空的关键词文件，确保与正式关键词库隔离
        for fname in ["blacklist_hard.txt", "blacklist_soft.txt"]:
            path = os.path.join(self.tmpdir, fname)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8"):
                    pass
        # 空文件已存在，重新加载本次测试专用的空关键词列表
        self.filter.hard_kw = []
        self.filter.soft_kw = []
        self.filter.blacklist_kw = []
        self.filter.watch_kw = []

    def teardown_method(self):
        """每个测试后清理"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_st_blocks_position(self, monkeypatch):
        """模拟ST列表返回600519，验证仓位被拦截"""
        def fake_st():
            return {"600519": "风险警示:贵州茅台"}
        monkeypatch.setattr(self.filter, "load_st_blacklist", fake_st)
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(self.filter, "scan_notice_keywords", lambda td, tc: {})

        picks = [{"code": "600519", "name": "贵州茅台", "position": 0.192, "action": "hold"}]
        result = self.filter.apply(picks, "2026-11-17", ["600519"])

        assert result[0]["risk"] == "BLACKLIST"
        assert result[0]["filtered_position"] == 0.0
        assert "风险警示" in result[0]["risk_reason"]

    def test_watch_halves_position(self, monkeypatch):
        """模拟公告命中软关键词，验证仓位减半"""
        monkeypatch.setattr(self.filter, "load_st_blacklist", lambda: {})
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(
            self.filter, "scan_notice_keywords",
            lambda td, tc: {"600519": {"hard": [], "soft": ["问询函"]}}
        )

        picks = [{"code": "600519", "name": "贵州茅台", "position": 0.192, "action": "hold"}]
        result = self.filter.apply(picks, "2026-11-17", ["600519"])

        assert result[0]["risk"] == "WATCH"
        assert abs(result[0]["filtered_position"] - 0.096) < 0.001
        assert "问询函" in result[0]["risk_reason"]

    def test_no_event_passthrough(self, monkeypatch):
        """无任何风险事件，验证仓位不变"""
        monkeypatch.setattr(self.filter, "load_st_blacklist", lambda: {})
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(self.filter, "scan_notice_keywords", lambda td, tc: {})

        picks = [
            {"code": "600519", "name": "贵州茅台", "position": 0.192, "action": "hold"},
            {"code": "000001", "name": "平安银行", "position": 0.160, "action": "hold"},
            {"code": "300750", "name": "宁德时代", "position": 0.350, "action": "buy"},
        ]
        result = self.filter.apply(picks, "2026-11-17", ["600519", "000001", "300750"])

        for row in result:
            assert row["risk"] == "NORMAL"
            assert abs(row["filtered_position"] - row["position_raw"]) < 0.0001
            assert row["risk_reason"] == ""

    def test_hard_keyword_blocks(self, monkeypatch):
        """模拟公告命中硬关键词，验证仓位强制0"""
        monkeypatch.setattr(self.filter, "load_st_blacklist", lambda: {})
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(
            self.filter, "scan_notice_keywords",
            lambda td, tc: {"300750": {"hard": ["立案调查"], "soft": []}}
        )

        picks = [{"code": "300750", "name": "宁德时代", "position": 0.350, "action": "buy"}]
        result = self.filter.apply(picks, "2026-11-17", ["300750"])

        assert result[0]["risk"] == "BLACKLIST"
        assert result[0]["filtered_position"] == 0.0
        assert "立案调查" in result[0]["risk_reason"]

    def test_st_overrides_notice(self, monkeypatch):
        """ST已拦截时，不叠加公告评级"""
        monkeypatch.setattr(
            self.filter, "load_st_blacklist",
            lambda: {"600519": "风险警示:贵州茅台"}
        )
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(
            self.filter, "scan_notice_keywords",
            lambda td, tc: {"600519": {"hard": ["立案调查"], "soft": []}}
        )

        picks = [{"code": "600519", "name": "贵州茅台", "position": 0.192, "action": "hold"}]
        result = self.filter.apply(picks, "2026-11-17", ["600519"])

        assert result[0]["risk"] == "BLACKLIST"
        assert "风险警示" in result[0]["risk_reason"]
        assert "立案调查" not in result[0]["risk_reason"]

    def test_csv_generated(self, monkeypatch):
        """验证 risk_blacklist.csv 正确写入"""
        monkeypatch.setattr(
            self.filter, "load_st_blacklist",
            lambda: {"600519": "风险警示:贵州茅台"}
        )
        monkeypatch.setattr(self.filter, "load_halt_list", lambda td: {})
        monkeypatch.setattr(self.filter, "scan_notice_keywords", lambda td, tc: {})

        picks = [{"code": "600519", "name": "贵州茅台", "position": 0.192, "action": "hold"}]
        self.filter.apply(picks, "2026-11-17", ["600519"])

        csv_path = os.path.join(self.tmpdir, "risk_blacklist.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 1
            assert rows[0]["code"] == "600519"
            assert rows[0]["risk_level"] == "BLACKLIST"


if __name__ == "__main__":
    """直接运行: python -m live.test_event_risk_filter"""
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
