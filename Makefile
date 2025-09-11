bk: freeze
	git add .
	git commit -m "backup"
	git push

cover:
	go test ./... -coverprofile=coverage.out
	go tool cover -func=coverage.out

freeze:
	pip freeze > requirements.txt
