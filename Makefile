lib:
	rm -rf build && mkdir build
	@if [ "$(IS_X86)" = "true" ]; then \
		arch -x86_64 /bin/bash -c "cd build && cmake ${CMAKE_FLAGS} .. && cmake --build . --config Release"; \
	else \
		cd build && cmake ${CMAKE_FLAGS} .. && cmake --build . --config Release ${CMAKE_BUILD_FLAGS}; \
	fi

.PHONY: clean
clean:
	rm -rf build

.PHONY: build
build:
	go build

.PHONY: test
test:
	go test -count 1 -v ./...