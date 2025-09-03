# 开发与测试文档

## 1. 开发环境设置

历史文本漂洗项目采用现代化的开发工具链和最佳实践，确保开发效率和代码质量。

### 1.1 开发环境架构

```mermaid
graph TB
    subgraph "开发者工作站"
        IDE["IDE/编辑器"]
        DOCKER["Docker Desktop"]
        GIT["Git客户端"]
        NODE["Node.js"]
        PYTHON["Python"]
    end
    
    subgraph "本地开发环境"
        LOCAL_DB["本地数据库"]
        LOCAL_REDIS["本地Redis"]
        LOCAL_ES["本地Elasticsearch"]
        DEV_SERVER["开发服务器"]
    end
    
    subgraph "开发工具"
        LINTER["代码检查"]
        FORMATTER["代码格式化"]
        DEBUGGER["调试器"]
        PROFILER["性能分析"]
    end
    
    subgraph "版本控制"
        GITLAB["GitLab"]
        HOOKS["Git Hooks"]
        CI_CD["CI/CD流水线"]
    end
    
    subgraph "测试环境"
        UNIT_TEST["单元测试"]
        INTEGRATION_TEST["集成测试"]
        E2E_TEST["端到端测试"]
        PERFORMANCE_TEST["性能测试"]
    end
    
    IDE --> DOCKER
    IDE --> GIT
    DOCKER --> LOCAL_DB
    DOCKER --> LOCAL_REDIS
    DOCKER --> LOCAL_ES
    
    GIT --> GITLAB
    GITLAB --> CI_CD
    CI_CD --> UNIT_TEST
    CI_CD --> INTEGRATION_TEST
    
    LINTER --> IDE
    FORMATTER --> IDE
    DEBUGGER --> IDE
```

### 1.2 开发环境配置

#### Docker Compose开发环境

```yaml
# docker-compose.dev.yml - 开发环境配置
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: historical_text_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    networks:
      - dev-network

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - dev-network

  # Elasticsearch搜索引擎
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    networks:
      - dev-network

  # MongoDB文档数据库
  mongodb:
    image: mongo:6
    environment:
      MONGO_INITDB_ROOT_USERNAME: dev_user
      MONGO_INITDB_ROOT_PASSWORD: dev_password
      MONGO_INITDB_DATABASE: historical_text_dev
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    networks:
      - dev-network

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: dev_user
      MINIO_ROOT_PASSWORD: dev_password
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    networks:
      - dev-network

  # Kafka消息队列
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - dev-network

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    networks:
      - dev-network

  # 开发工具
  mailhog:
    image: mailhog/mailhog:latest
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    networks:
      - dev-network

  # 数据库管理工具
  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    networks:
      - dev-network

  # Redis管理工具
  redis-commander:
    image: rediscommander/redis-commander:latest
    environment:
      REDIS_HOSTS: local:redis:6379
    ports:
      - "8081:8081"
    networks:
      - dev-network

volumes:
  postgres_data:
  redis_data:
  es_data:
  mongo_data:
  minio_data:

networks:
  dev-network:
    driver: bridge
```

#### 开发环境启动脚本

```bash
#!/bin/bash
# scripts/dev-setup.sh - 开发环境设置脚本

set -e

echo "🚀 Setting up development environment..."

# 检查依赖
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }

# 创建必要的目录
mkdir -p logs
mkdir -p uploads
mkdir -p temp

# 复制环境配置文件
if [ ! -f .env.dev ]; then
    cp .env.example .env.dev
    echo "📝 Created .env.dev file. Please update it with your settings."
fi

# 启动基础设施服务
echo "🐳 Starting infrastructure services..."
docker-compose -f docker-compose.dev.yml up -d

# 等待服务启动
echo "⏳ Waiting for services to be ready..."
sleep 30

# 检查服务状态
echo "🔍 Checking service status..."
docker-compose -f docker-compose.dev.yml ps

# 初始化数据库
echo "🗄️ Initializing database..."
python3 scripts/init-database.py

# 安装前端依赖
if [ -d "frontend" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# 安装后端依赖
if [ -f "requirements.txt" ]; then
    echo "🐍 Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
fi

# 运行数据库迁移
echo "🔄 Running database migrations..."
python3 manage.py migrate

# 创建超级用户（如果不存在）
echo "👤 Creating superuser..."
python3 scripts/create-superuser.py

echo "✅ Development environment setup complete!"
echo ""
echo "🌐 Available services:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo "  - Elasticsearch: http://localhost:9200"
echo "  - MongoDB: localhost:27017"
echo "  - MinIO: http://localhost:9001"
echo "  - MailHog: http://localhost:8025"
echo "  - PgAdmin: http://localhost:5050"
echo "  - Redis Commander: http://localhost:8081"
echo ""
echo "🚀 To start development servers:"
echo "  Frontend: cd frontend && npm run dev"
echo "  Backend: python3 manage.py runserver"
```

### 1.3 IDE配置

#### VS Code配置

```json
// .vscode/settings.json - VS Code设置
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  
  "typescript.preferences.importModuleSpecifier": "relative",
  "typescript.suggest.autoImports": true,
  "typescript.updateImportsOnFileMove.enabled": "always",
  
  "eslint.workingDirectories": ["frontend"],
  "eslint.validate": ["javascript", "typescript", "vue"],
  
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll.eslint": true
  },
  
  "files.associations": {
    "*.vue": "vue"
  },
  
  "emmet.includeLanguages": {
    "vue-html": "html"
  },
  
  "git.autofetch": true,
  "git.enableSmartCommit": true,
  
  "docker.showStartPage": false,
  
  "yaml.schemas": {
    "https://json.schemastore.org/github-workflow.json": ".github/workflows/*.yml",
    "https://json.schemastore.org/docker-compose.json": "docker-compose*.yml"
  }
}

// .vscode/extensions.json - 推荐扩展
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-python.black-formatter",
    "ms-python.isort",
    "Vue.volar",
    "Vue.vscode-typescript-vue-plugin",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "ms-vscode-remote.remote-containers",
    "ms-azuretools.vscode-docker",
    "GitLab.gitlab-workflow",
    "redhat.vscode-yaml",
    "ms-vscode.test-adapter-converter",
    "hbenl.vscode-test-explorer"
  ]
}

// .vscode/launch.json - 调试配置
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/backend/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/backend"
      },
      "args": ["--reload", "--host", "0.0.0.0", "--port", "8000"]
    },
    {
      "name": "Python: Django",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["runserver", "0.0.0.0:8000"],
      "django": true,
      "console": "integratedTerminal"
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    },
    {
      "name": "Python: Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${workspaceFolder}/tests", "-v"],
      "console": "integratedTerminal"
    }
  ]
}

// .vscode/tasks.json - 任务配置
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Dev Environment",
      "type": "shell",
      "command": "./scripts/dev-setup.sh",
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "python",
      "args": ["-m", "pytest", "tests/", "-v"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Format Code",
      "type": "shell",
      "command": "black",
      "args": ["."],
      "group": "build"
    },
    {
      "label": "Lint Code",
      "type": "shell",
      "command": "flake8",
      "args": ["."],
      "group": "build"
    },
    {
      "label": "Type Check",
      "type": "shell",
      "command": "mypy",
      "args": ["."],
      "group": "build"
    }
  ]
}
```

## 2. 代码质量保证

### 2.1 代码规范配置

#### Python代码规范

```ini
# setup.cfg - Python工具配置
[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503
max-complexity = 10
select = B,C,E,F,W,T4,B9
exclude = 
    .git,
    __pycache__,
    .venv,
    venv,
    migrations,
    build,
    dist

[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    external: Tests that require external services

[coverage:run]
source = src
omit = 
    */tests/*
    */migrations/*
    */venv/*
    */env/*
    manage.py
    */settings/*
    */wsgi.py
    */asgi.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\(Protocol\):
    @(abc\.)?abstractmethod
```

```toml
# pyproject.toml - 现代Python配置
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "historical-text-processing"
version = "1.0.0"
description = "Historical Text Processing and Analysis System"
authors = [{name = "Development Team", email = "dev@example.com"}]
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "pydantic>=2.4.0",
    "redis>=5.0.0",
    "celery>=5.3.0",
    "elasticsearch>=8.8.0",
    "pymongo>=4.5.0",
    "boto3>=1.28.0",
    "pillow>=10.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    "bandit>=1.7.0",
    "safety>=2.3.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",
    "httpx>=0.24.0",
    "factory-boy>=3.3.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.1.0",
    "mkdocstrings[python]>=0.22.0",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
(
  /(
      \.eggs
    | \.git
    | \.hg
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    | dist
    | migrations
  )/
)
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
skip_glob = ["*/migrations/*"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow tests",
    "external: Tests that require external services",
]

[tool.bandit]
exclude_dirs = ["tests", "migrations"]
skips = ["B101", "B601"]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/venv/*",
    "*/env/*",
    "manage.py",
    "*/settings/*",
    "*/wsgi.py",
    "*/asgi.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "class .*\\(Protocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

#### 前端代码规范

```json
// frontend/.eslintrc.js - ESLint配置
module.exports = {
  root: true,
  env: {
    node: true,
    browser: true,
    es2022: true
  },
  extends: [
    'plugin:vue/vue3-essential',
    'plugin:vue/vue3-strongly-recommended',
    'plugin:vue/vue3-recommended',
    '@vue/eslint-config-typescript',
    '@vue/eslint-config-prettier'
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'vue/multi-word-component-names': 'off',
    'vue/no-unused-vars': 'error',
    'vue/no-unused-components': 'error',
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    'prefer-const': 'error',
    'no-var': 'error'
  },
  overrides: [
    {
      files: ['**/__tests__/*.{j,t}s?(x)', '**/tests/unit/**/*.spec.{j,t}s?(x)'],
      env: {
        jest: true
      }
    }
  ]
}

// frontend/.prettierrc.js - Prettier配置
module.exports = {
  semi: false,
  singleQuote: true,
  quoteProps: 'as-needed',
  trailingComma: 'es5',
  bracketSpacing: true,
  bracketSameLine: false,
  arrowParens: 'avoid',
  printWidth: 80,
  tabWidth: 2,
  useTabs: false,
  endOfLine: 'lf',
  vueIndentScriptAndStyle: false
}

// frontend/tsconfig.json - TypeScript配置
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@/components/*": ["src/components/*"],
      "@/views/*": ["src/views/*"],
      "@/utils/*": ["src/utils/*"],
      "@/api/*": ["src/api/*"],
      "@/stores/*": ["src/stores/*"]
    },
    "types": ["vite/client", "node", "jest"]
  },
  "include": [
    "src/**/*.ts",
    "src/**/*.d.ts",
    "src/**/*.tsx",
    "src/**/*.vue",
    "tests/**/*.ts",
    "tests/**/*.tsx"
  ],
  "exclude": ["node_modules", "dist"]
}
```

### 2.2 Git Hooks配置

```yaml
# .pre-commit-config.yaml - Pre-commit配置
repos:
  # Python代码格式化和检查
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  # 安全检查
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  # 通用检查
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-xml
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: check-docstring-first
      - id: debug-statements
      - id: requirements-txt-fixer

  # 前端代码检查
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.47.0
    hooks:
      - id: eslint
        files: \.(js|ts|vue)$
        types: [file]
        additional_dependencies:
          - eslint@8.47.0
          - "@vue/eslint-config-typescript@11.0.3"
          - "@vue/eslint-config-prettier@8.0.0"
          - "eslint-plugin-vue@9.17.0"

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.1
    hooks:
      - id: prettier
        files: \.(js|ts|vue|json|css|scss|md)$

  # Docker文件检查
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint

  # 密钥检查
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # 提交信息检查
  - repo: https://github.com/commitizen-tools/commitizen
    rev: 3.6.0
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

```bash
#!/bin/bash
# .githooks/pre-push - 推送前检查

set -e

echo "🔍 Running pre-push checks..."

# 运行测试
echo "🧪 Running tests..."
python -m pytest tests/ -x --tb=short

# 检查测试覆盖率
echo "📊 Checking test coverage..."
python -m pytest tests/ --cov=src --cov-fail-under=80

# 安全检查
echo "🔒 Running security checks..."
bandit -r src/ -f json -o bandit-report.json
safety check --json --output safety-report.json

# 依赖检查
echo "📦 Checking dependencies..."
pip-audit --format=json --output=pip-audit-report.json

# 前端检查（如果存在）
if [ -d "frontend" ]; then
    echo "🎨 Running frontend checks..."
    cd frontend
    npm run lint
    npm run type-check
    npm run test:unit
    cd ..
fi

echo "✅ All pre-push checks passed!"
```

## 3. 测试策略

### 3.1 测试金字塔

```mermaid
pyramid
    title 测试金字塔
    "E2E Tests" : 10
    "Integration Tests" : 20
    "Unit Tests" : 70
```

### 3.2 单元测试

```python
# tests/unit/test_document_service.py - 单元测试示例
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from src.services.document_service import DocumentService
from src.models.document import Document, DocumentStatus
from src.exceptions import DocumentNotFoundError, ProcessingError


class TestDocumentService:
    """文档服务单元测试"""
    
    @pytest.fixture
    def mock_db_session(self):
        """模拟数据库会话"""
        return Mock()
    
    @pytest.fixture
    def mock_storage_service(self):
        """模拟存储服务"""
        return Mock()
    
    @pytest.fixture
    def mock_ocr_service(self):
        """模拟OCR服务"""
        return Mock()
    
    @pytest.fixture
    def document_service(self, mock_db_session, mock_storage_service, mock_ocr_service):
        """文档服务实例"""
        return DocumentService(
            db_session=mock_db_session,
            storage_service=mock_storage_service,
            ocr_service=mock_ocr_service
        )
    
    @pytest.fixture
    def sample_document(self) -> Document:
        """示例文档"""
        return Document(
            id="doc-123",
            title="测试文档",
            filename="test.pdf",
            file_size=1024,
            mime_type="application/pdf",
            status=DocumentStatus.UPLOADED,
            created_at=datetime.utcnow(),
            user_id="user-123"
        )
    
    async def test_create_document_success(self, document_service, mock_db_session):
        """测试成功创建文档"""
        # Arrange
        document_data = {
            "title": "新文档",
            "filename": "new.pdf",
            "file_size": 2048,
            "mime_type": "application/pdf",
            "user_id": "user-456"
        }
        
        mock_db_session.add = Mock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()
        
        # Act
        result = await document_service.create_document(document_data)
        
        # Assert
        assert result.title == document_data["title"]
        assert result.filename == document_data["filename"]
        assert result.status == DocumentStatus.UPLOADED
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    async def test_get_document_success(self, document_service, mock_db_session, sample_document):
        """测试成功获取文档"""
        # Arrange
        document_id = "doc-123"
        mock_db_session.get = AsyncMock(return_value=sample_document)
        
        # Act
        result = await document_service.get_document(document_id)
        
        # Assert
        assert result == sample_document
        mock_db_session.get.assert_called_once_with(Document, document_id)
    
    async def test_get_document_not_found(self, document_service, mock_db_session):
        """测试文档不存在"""
        # Arrange
        document_id = "non-existent"
        mock_db_session.get = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(DocumentNotFoundError) as exc_info:
            await document_service.get_document(document_id)
        
        assert str(exc_info.value) == f"Document {document_id} not found"
    
    @patch('src.services.document_service.celery_app')
    async def test_process_document_success(self, mock_celery, document_service, sample_document):
        """测试成功处理文档"""
        # Arrange
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_celery.send_task.return_value = mock_task
        
        # Act
        task_id = await document_service.process_document(sample_document.id)
        
        # Assert
        assert task_id == "task-123"
        mock_celery.send_task.assert_called_once_with(
            'process_document_task',
            args=[sample_document.id]
        )
    
    async def test_ocr_processing_success(self, document_service, mock_ocr_service, sample_document):
        """测试OCR处理成功"""
        # Arrange
        expected_text = "这是提取的文本内容"
        mock_ocr_service.extract_text = AsyncMock(return_value=expected_text)
        
        # Act
        result = await document_service.extract_text(sample_document)
        
        # Assert
        assert result == expected_text
        mock_ocr_service.extract_text.assert_called_once_with(sample_document)
    
    async def test_ocr_processing_failure(self, document_service, mock_ocr_service, sample_document):
        """测试OCR处理失败"""
        # Arrange
        mock_ocr_service.extract_text = AsyncMock(side_effect=Exception("OCR failed"))
        
        # Act & Assert
        with pytest.raises(ProcessingError) as exc_info:
            await document_service.extract_text(sample_document)
        
        assert "OCR processing failed" in str(exc_info.value)
    
    @pytest.mark.parametrize("file_size,expected_valid", [
        (1024, True),
        (10 * 1024 * 1024, True),  # 10MB
        (100 * 1024 * 1024, False),  # 100MB
        (0, False),
    ])
    def test_validate_file_size(self, document_service, file_size, expected_valid):
        """测试文件大小验证"""
        # Act
        result = document_service.validate_file_size(file_size)
        
        # Assert
        assert result == expected_valid
    
    @pytest.mark.parametrize("mime_type,expected_valid", [
        ("application/pdf", True),
        ("image/jpeg", True),
        ("image/png", True),
        ("text/plain", False),
        ("application/zip", False),
    ])
    def test_validate_mime_type(self, document_service, mime_type, expected_valid):
        """测试MIME类型验证"""
        # Act
        result = document_service.validate_mime_type(mime_type)
        
        # Assert
        assert result == expected_valid


# tests/unit/test_user_service.py - 用户服务测试
import pytest
from unittest.mock import Mock, AsyncMock, patch
from passlib.context import CryptContext

from src.services.user_service import UserService
from src.models.user import User, UserRole
from src.exceptions import UserNotFoundError, InvalidCredentialsError


class TestUserService:
    """用户服务单元测试"""
    
    @pytest.fixture
    def mock_db_session(self):
        return Mock()
    
    @pytest.fixture
    def mock_redis_client(self):
        return Mock()
    
    @pytest.fixture
    def user_service(self, mock_db_session, mock_redis_client):
        return UserService(
            db_session=mock_db_session,
            redis_client=mock_redis_client
        )
    
    @pytest.fixture
    def sample_user(self) -> User:
        return User(
            id="user-123",
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$hashed_password",
            role=UserRole.USER,
            is_active=True
        )
    
    async def test_authenticate_user_success(self, user_service, mock_db_session, sample_user):
        """测试用户认证成功"""
        # Arrange
        username = "testuser"
        password = "password123"
        
        mock_db_session.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db_session.execute.return_value = mock_result
        
        with patch.object(user_service.pwd_context, 'verify', return_value=True):
            # Act
            result = await user_service.authenticate_user(username, password)
            
            # Assert
            assert result == sample_user
    
    async def test_authenticate_user_invalid_password(self, user_service, mock_db_session, sample_user):
        """测试密码错误"""
        # Arrange
        username = "testuser"
        password = "wrongpassword"
        
        mock_db_session.execute = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db_session.execute.return_value = mock_result
        
        with patch.object(user_service.pwd_context, 'verify', return_value=False):
            # Act & Assert
            with pytest.raises(InvalidCredentialsError):
                await user_service.authenticate_user(username, password)
    
    async def test_create_user_success(self, user_service, mock_db_session):
        """测试创建用户成功"""
        # Arrange
        user_data = {
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123"
        }
        
        mock_db_session.add = Mock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()
        
        with patch.object(user_service.pwd_context, 'hash', return_value="hashed_password"):
            # Act
            result = await user_service.create_user(user_data)
            
            # Assert
            assert result.username == user_data["username"]
            assert result.email == user_data["email"]
            assert result.role == UserRole.USER
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
```

### 3.3 集成测试

```python
# tests/integration/test_document_api.py - API集成测试
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from src.main import app
from src.database import get_db_session
from src.models.base import Base
from src.models.user import User
from src.models.document import Document
from tests.factories import UserFactory, DocumentFactory


# 测试数据库配置
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_engine):
    """创建测试数据库会话"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_client(test_db_session):
    """创建测试客户端"""
    def override_get_db():
        return test_db_session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(test_db_session) -> User:
    """创建测试用户"""
    user = UserFactory.build()
    test_db_session.add(user)
    await test_db_session.commit()
    await test_db_session.refresh(user)
    return user


@pytest.fixture
async def auth_headers(test_client, test_user) -> dict:
    """获取认证头"""
    login_data = {
        "username": test_user.username,
        "password": "password123"  # 默认密码
    }
    
    response = await test_client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestDocumentAPI:
    """文档API集成测试"""
    
    async def test_upload_document_success(self, test_client, auth_headers):
        """测试文档上传成功"""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"PDF content")
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as file:
                files = {"file": ("test.pdf", file, "application/pdf")}
                data = {"title": "测试文档"}
                
                # Act
                response = await test_client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data,
                    headers=auth_headers
                )
                
                # Assert
                assert response.status_code == 201
                result = response.json()
                assert result["title"] == "测试文档"
                assert result["filename"] == "test.pdf"
                assert result["status"] == "uploaded"
        finally:
            os.unlink(tmp_file_path)
    
    async def test_upload_document_invalid_file_type(self, test_client, auth_headers):
        """测试上传无效文件类型"""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"Text content")
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as file:
                files = {"file": ("test.txt", file, "text/plain")}
                data = {"title": "测试文档"}
                
                # Act
                response = await test_client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data,
                    headers=auth_headers
                )
                
                # Assert
                assert response.status_code == 400
                assert "Invalid file type" in response.json()["detail"]
        finally:
            os.unlink(tmp_file_path)
    
    async def test_get_document_success(self, test_client, auth_headers, test_db_session, test_user):
        """测试获取文档成功"""
        # Arrange
        document = DocumentFactory.build(user_id=test_user.id)
        test_db_session.add(document)
        await test_db_session.commit()
        await test_db_session.refresh(document)
        
        # Act
        response = await test_client.get(
            f"/api/v1/documents/{document.id}",
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == document.id
        assert result["title"] == document.title
    
    async def test_get_document_not_found(self, test_client, auth_headers):
        """测试文档不存在"""
        # Act
        response = await test_client.get(
            "/api/v1/documents/non-existent",
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 404
    
    async def test_list_documents_success(self, test_client, auth_headers, test_db_session, test_user):
        """测试文档列表获取成功"""
        # Arrange
        documents = [DocumentFactory.build(user_id=test_user.id) for _ in range(3)]
        for doc in documents:
            test_db_session.add(doc)
        await test_db_session.commit()
        
        # Act
        response = await test_client.get(
            "/api/v1/documents/",
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert len(result["items"]) == 3
        assert result["total"] == 3
    
    async def test_process_document_success(self, test_client, auth_headers, test_db_session, test_user):
        """测试文档处理成功"""
        # Arrange
        document = DocumentFactory.build(user_id=test_user.id)
        test_db_session.add(document)
        await test_db_session.commit()
        await test_db_session.refresh(document)
        
        # Act
        response = await test_client.post(
            f"/api/v1/documents/{document.id}/process",
            headers=auth_headers
        )
        
        # Assert
        assert response.status_code == 202
        result = response.json()
        assert "task_id" in result
    
    async def test_unauthorized_access(self, test_client):
        """测试未授权访问"""
        # Act
        response = await test_client.get("/api/v1/documents/")
        
        # Assert
        assert response.status_code == 401


class TestUserAPI:
    """用户API集成测试"""
    
    async def test_register_user_success(self, test_client):
        """测试用户注册成功"""
        # Arrange
        user_data = {
            "username": "newuser",
            "email": "new@example.com",
            "password": "password123"
        }
        
        # Act
        response = await test_client.post("/api/v1/auth/register", json=user_data)
        
        # Assert
        assert response.status_code == 201
        result = response.json()
        assert result["username"] == user_data["username"]
        assert result["email"] == user_data["email"]
        assert "password" not in result
    
    async def test_login_success(self, test_client, test_user):
        """测试登录成功"""
        # Arrange
        login_data = {
            "username": test_user.username,
            "password": "password123"
        }
        
        # Act
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        
        # Assert
        assert response.status_code == 200
        result = response.json()
        assert "access_token" in result
        assert result["token_type"] == "bearer"
    
    async def test_login_invalid_credentials(self, test_client, test_user):
        """测试登录凭据无效"""
        # Arrange
        login_data = {
            "username": test_user.username,
            "password": "wrongpassword"
        }
        
        # Act
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        
        # Assert
        assert response.status_code == 401
```

### 3.4 端到端测试

```python
# tests/e2e/test_document_workflow.py - 端到端测试
import pytest
from playwright.async_api import async_playwright, Page, Browser
import asyncio
import tempfile
import os
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def browser() -> AsyncGenerator[Browser, None]:
    """启动浏览器"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser) -> AsyncGenerator[Page, None]:
    """创建页面"""
    page = await browser.new_page()
    yield page
    await page.close()


@pytest.fixture
async def authenticated_page(page: Page) -> Page:
    """已认证的页面"""
    # 导航到登录页面
    await page.goto("http://localhost:3000/login")
    
    # 填写登录表单
    await page.fill('[data-testid="username-input"]', "testuser")
    await page.fill('[data-testid="password-input"]', "password123")
    await page.click('[data-testid="login-button"]')
    
    # 等待登录完成
    await page.wait_for_url("http://localhost:3000/dashboard")
    
    return page


class TestDocumentWorkflow:
    """文档处理工作流端到端测试"""
    
    async def test_complete_document_processing_workflow(self, authenticated_page: Page):
        """测试完整的文档处理工作流"""
        page = authenticated_page
        
        # 1. 导航到文档上传页面
        await page.click('[data-testid="upload-nav"]')
        await page.wait_for_url("http://localhost:3000/upload")
        
        # 2. 创建测试文件
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"Test PDF content")
            tmp_file_path = tmp_file.name
        
        try:
            # 3. 上传文件
            await page.set_input_files('[data-testid="file-input"]', tmp_file_path)
            await page.fill('[data-testid="title-input"]', "测试文档")
            await page.click('[data-testid="upload-button"]')
            
            # 4. 等待上传完成
            await page.wait_for_selector('[data-testid="upload-success"]')
            
            # 5. 导航到文档列表
            await page.click('[data-testid="documents-nav"]')
            await page.wait_for_url("http://localhost:3000/documents")
            
            # 6. 验证文档出现在列表中
            await page.wait_for_selector('[data-testid="document-item"]')
            document_title = await page.text_content('[data-testid="document-title"]')
            assert document_title == "测试文档"
            
            # 7. 点击处理按钮
            await page.click('[data-testid="process-button"]')
            
            # 8. 等待处理开始
            await page.wait_for_selector('[data-testid="processing-status"]')
            
            # 9. 等待处理完成（最多等待30秒）
            await page.wait_for_selector(
                '[data-testid="processing-complete"]',
                timeout=30000
            )
            
            # 10. 验证处理结果
            await page.click('[data-testid="view-result-button"]')
            await page.wait_for_selector('[data-testid="extracted-text"]')
            
            extracted_text = await page.text_content('[data-testid="extracted-text"]')
            assert len(extracted_text) > 0
            
        finally:
            # 清理测试文件
            os.unlink(tmp_file_path)
    
    async def test_search_functionality(self, authenticated_page: Page):
        """测试搜索功能"""
        page = authenticated_page
        
        # 1. 导航到搜索页面
        await page.click('[data-testid="search-nav"]')
        await page.wait_for_url("http://localhost:3000/search")
        
        # 2. 输入搜索关键词
        await page.fill('[data-testid="search-input"]', "历史文本")
        await page.click('[data-testid="search-button"]')
        
        # 3. 等待搜索结果
        await page.wait_for_selector('[data-testid="search-results"]')
        
        # 4. 验证搜索结果
        results = await page.query_selector_all('[data-testid="search-result-item"]')
        assert len(results) > 0
        
        # 5. 点击第一个结果
        await results[0].click()
        
        # 6. 验证详情页面
        await page.wait_for_selector('[data-testid="document-detail"]')
    
    async def test_user_profile_management(self, authenticated_page: Page):
        """测试用户资料管理"""
        page = authenticated_page
        
        # 1. 导航到用户资料页面
        await page.click('[data-testid="profile-nav"]')
        await page.wait_for_url("http://localhost:3000/profile")
        
        # 2. 编辑用户信息
        await page.click('[data-testid="edit-profile-button"]')
        await page.fill('[data-testid="display-name-input"]', "新的显示名称")
        await page.click('[data-testid="save-profile-button"]')
        
        # 3. 验证保存成功
        await page.wait_for_selector('[data-testid="save-success"]')
        
        # 4. 验证信息更新
        display_name = await page.text_content('[data-testid="display-name"]')
        assert display_name == "新的显示名称"


## 4. 性能测试

### 4.1 负载测试配置

```python
# tests/performance/test_load.py - 负载测试
import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import statistics
import json


class LoadTestRunner:
    """负载测试运行器"""
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.results: List[Dict[str, Any]] = []
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, method: str = 'GET', data: dict = None) -> Dict[str, Any]:
        """发送HTTP请求"""
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        start_time = time.time()
        
        try:
            async with session.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
                headers=headers,
                json=data
            ) as response:
                await response.text()
                end_time = time.time()
                
                return {
                    'endpoint': endpoint,
                    'method': method,
                    'status_code': response.status,
                    'response_time': end_time - start_time,
                    'success': 200 <= response.status < 400
                }
        except Exception as e:
            end_time = time.time()
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': 0,
                'response_time': end_time - start_time,
                'success': False,
                'error': str(e)
            }
    
    async def run_concurrent_requests(self, endpoint: str, concurrent_users: int, requests_per_user: int, method: str = 'GET', data: dict = None):
        """运行并发请求测试"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    task = self.make_request(session, endpoint, method, data)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            self.results.extend(results)
            
            return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.results:
            return {}
        
        response_times = [r['response_time'] for r in self.results]
        success_count = sum(1 for r in self.results if r['success'])
        total_requests = len(self.results)
        
        return {
            'total_requests': total_requests,
            'successful_requests': success_count,
            'failed_requests': total_requests - success_count,
            'success_rate': success_count / total_requests * 100,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'p50_response_time': statistics.median(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        }
    
    def generate_report(self) -> str:
        """生成测试报告"""
        analysis = self.analyze_results()
        
        report = f"""
负载测试报告
=============

总请求数: {analysis['total_requests']}
成功请求数: {analysis['successful_requests']}
失败请求数: {analysis['failed_requests']}
成功率: {analysis['success_rate']:.2f}%

响应时间统计:
- 平均响应时间: {analysis['avg_response_time']:.3f}s
- 最小响应时间: {analysis['min_response_time']:.3f}s
- 最大响应时间: {analysis['max_response_time']:.3f}s
- P50响应时间: {analysis['p50_response_time']:.3f}s
- P95响应时间: {analysis['p95_response_time']:.3f}s
- P99响应时间: {analysis['p99_response_time']:.3f}s
"""
        
        return report


# 测试场景
async def test_api_endpoints_load():
    """API端点负载测试"""
    runner = LoadTestRunner('http://localhost:8000')
    
    # 测试场景1: 文档列表API
    print("测试文档列表API...")
    await runner.run_concurrent_requests(
        endpoint='/api/v1/documents/',
        concurrent_users=50,
        requests_per_user=10
    )
    
    # 测试场景2: 搜索API
    print("测试搜索API...")
    await runner.run_concurrent_requests(
        endpoint='/api/v1/search?q=历史文本',
        concurrent_users=30,
        requests_per_user=20
    )
    
    # 测试场景3: 用户认证API
    print("测试用户认证API...")
    await runner.run_concurrent_requests(
        endpoint='/api/v1/auth/login',
        method='POST',
        data={'username': 'testuser', 'password': 'password123'},
        concurrent_users=20,
        requests_per_user=5
    )
    
    # 生成报告
    report = runner.generate_report()
    print(report)
    
    # 保存详细结果
    with open('load_test_results.json', 'w') as f:
        json.dump(runner.results, f, indent=2)


if __name__ == '__main__':
    asyncio.run(test_api_endpoints_load())
```

### 4.2 压力测试脚本

```bash
#!/bin/bash
# scripts/stress-test.sh - 压力测试脚本

set -e

echo "🔥 Starting stress test..."

# 配置
BASE_URL="http://localhost:8000"
MAX_USERS=1000
RAMP_UP_TIME=300  # 5分钟
TEST_DURATION=600  # 10分钟

# 检查依赖
command -v artillery >/dev/null 2>&1 || { echo "❌ Artillery is required but not installed."; exit 1; }

# 创建Artillery配置
cat > artillery-config.yml << EOF
config:
  target: '${BASE_URL}'
  phases:
    - duration: ${RAMP_UP_TIME}
      arrivalRate: 1
      rampTo: 50
      name: "Ramp up"
    - duration: ${TEST_DURATION}
      arrivalRate: 50
      name: "Sustained load"
  processor: "./artillery-processor.js"
  variables:
    auth_token: "your-test-token"

scenarios:
  - name: "Document API Load Test"
    weight: 40
    flow:
      - get:
          url: "/api/v1/documents/"
          headers:
            Authorization: "Bearer {{ auth_token }}"
      - think: 2
      - get:
          url: "/api/v1/documents/{{ \$randomString() }}"
          headers:
            Authorization: "Bearer {{ auth_token }}"
          expect:
            - statusCode: [200, 404]
  
  - name: "Search API Load Test"
    weight: 30
    flow:
      - get:
          url: "/api/v1/search"
          qs:
            q: "{{ \$randomString() }}"
          headers:
            Authorization: "Bearer {{ auth_token }}"
      - think: 1
  
  - name: "User Authentication Test"
    weight: 20
    flow:
      - post:
          url: "/api/v1/auth/login"
          json:
            username: "testuser{{ \$randomInt(1, 100) }}"
            password: "password123"
      - think: 3
  
  - name: "File Upload Test"
    weight: 10
    flow:
      - post:
          url: "/api/v1/documents/upload"
          headers:
            Authorization: "Bearer {{ auth_token }}"
          formData:
            file: "@test-file.pdf"
            title: "Test Document {{ \$randomString() }}"
EOF

# 创建处理器脚本
cat > artillery-processor.js << 'EOF'
module.exports = {
  generateRandomString: generateRandomString,
  generateRandomInt: generateRandomInt
};

function generateRandomString(context, events, done) {
  const strings = ['历史文本', '古代文献', '文档处理', '文本分析', '数据挖掘'];
  context.vars.randomString = strings[Math.floor(Math.random() * strings.length)];
  return done();
}

function generateRandomInt(context, events, done) {
  context.vars.randomInt = Math.floor(Math.random() * 1000) + 1;
  return done();
}
EOF

# 创建测试文件
echo "Creating test file..."
dd if=/dev/zero of=test-file.pdf bs=1024 count=100

# 运行压力测试
echo "🚀 Running stress test with Artillery..."
artillery run artillery-config.yml --output stress-test-report.json

# 生成HTML报告
echo "📊 Generating HTML report..."
artillery report stress-test-report.json --output stress-test-report.html

# 清理
rm -f artillery-config.yml artillery-processor.js test-file.pdf

echo "✅ Stress test completed!"
echo "📈 Report saved to: stress-test-report.html"
```

## 5. CI/CD流水线

### 5.1 GitLab CI配置

```yaml
# .gitlab-ci.yml - GitLab CI/CD配置
stages:
  - validate
  - test
  - build
  - security
  - deploy
  - performance

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  POSTGRES_DB: test_db
  POSTGRES_USER: test_user
  POSTGRES_PASSWORD: test_password
  REDIS_URL: redis://redis:6379

# 代码质量检查
code-quality:
  stage: validate
  image: python:3.11
  before_script:
    - pip install -r requirements-dev.txt
  script:
    - black --check .
    - isort --check-only .
    - flake8 .
    - mypy .
    - bandit -r src/
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 前端代码检查
frontend-lint:
  stage: validate
  image: node:18
  before_script:
    - cd frontend
    - npm ci
  script:
    - npm run lint
    - npm run type-check
  cache:
    paths:
      - frontend/node_modules/
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 单元测试
unit-tests:
  stage: test
  image: python:3.11
  services:
    - postgres:15
    - redis:7
  variables:
    DATABASE_URL: postgresql://test_user:test_password@postgres:5432/test_db
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=term
  coverage: '/TOTAL.+ ([0-9]{1,3}%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 集成测试
integration-tests:
  stage: test
  image: python:3.11
  services:
    - postgres:15
    - redis:7
    - elasticsearch:8.8.0
  variables:
    DATABASE_URL: postgresql://test_user:test_password@postgres:5432/test_db
    ELASTICSEARCH_URL: http://elasticsearch:9200
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - python manage.py migrate
  script:
    - pytest tests/integration/ -v --tb=short
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 前端测试
frontend-tests:
  stage: test
  image: node:18
  before_script:
    - cd frontend
    - npm ci
  script:
    - npm run test:unit
    - npm run test:e2e:headless
  artifacts:
    reports:
      junit: frontend/test-results.xml
    paths:
      - frontend/coverage/
    expire_in: 1 week
  cache:
    paths:
      - frontend/node_modules/
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 构建后端镜像
build-backend:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" == "$CI_DEFAULT_BRANCH" ]; then
        docker tag $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE/backend:latest
        docker push $CI_REGISTRY_IMAGE/backend:latest
      fi
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# 构建前端镜像
build-frontend:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -f frontend/Dockerfile -t $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA frontend/
    - docker push $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" == "$CI_DEFAULT_BRANCH" ]; then
        docker tag $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE/frontend:latest
        docker push $CI_REGISTRY_IMAGE/frontend:latest
      fi
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

# 安全扫描
security-scan:
  stage: security
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock 
        aquasec/trivy image --exit-code 0 --severity HIGH,CRITICAL 
        $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock 
        aquasec/trivy image --exit-code 0 --severity HIGH,CRITICAL 
        $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA
  allow_failure: true
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 部署到开发环境
deploy-dev:
  stage: deploy
  image: bitnami/kubectl:latest
  before_script:
    - echo $KUBE_CONFIG | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    - kubectl set image deployment/backend-deployment backend=$CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA -n historical-text-dev
    - kubectl set image deployment/frontend-deployment frontend=$CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA -n historical-text-dev
    - kubectl rollout status deployment/backend-deployment -n historical-text-dev
    - kubectl rollout status deployment/frontend-deployment -n historical-text-dev
  environment:
    name: development
    url: https://dev.historical-text.com
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"

# 部署到生产环境
deploy-prod:
  stage: deploy
  image: bitnami/kubectl:latest
  before_script:
    - echo $KUBE_CONFIG_PROD | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
  script:
    - kubectl set image deployment/backend-deployment backend=$CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA -n historical-text-prod
    - kubectl set image deployment/frontend-deployment frontend=$CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA -n historical-text-prod
    - kubectl rollout status deployment/backend-deployment -n historical-text-prod
    - kubectl rollout status deployment/frontend-deployment -n historical-text-prod
  environment:
    name: production
    url: https://historical-text.com
  when: manual
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 性能测试
performance-test:
  stage: performance
  image: node:18
  before_script:
    - npm install -g artillery
  script:
    - artillery run tests/performance/load-test.yml --output performance-report.json
    - artillery report performance-report.json --output performance-report.html
  artifacts:
    paths:
      - performance-report.html
      - performance-report.json
    expire_in: 1 week
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
      when: manual
```

---

**文档版本**: v1.0  
**最后更新**: 2024-01-01  
**维护者**: 开发团队